#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from collections import Counter
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

 # Load the pre-trained word2vec_model
word2vec_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

# Define a function to preprocess data so it can be called in the prediction step.
def preprocess(path='./combined_application_data_2021-23.csv'):
    # Read data
    all_staff = pd.read_csv(path)

    # Column Combination
    all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?'].fillna('', inplace=True)
    all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?2'].fillna('', inplace=True)
    all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?3'].fillna('', inplace=True)

    all_staff['familiar_tech'] = all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?'] +                         ' ' + all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?2'] +                         ' ' + all_staff['Which of the following programming languages, databases, and services are you familiar with and have experience using?3']

    all_staff['familiar_tech']

    columns_to_drop = ['Which of the following programming languages, databases, and services are you familiar with and have experience using?',
                    'Which of the following programming languages, databases, and services are you familiar with and have experience using?2',
                    'Which of the following programming languages, databases, and services are you familiar with and have experience using?3']

    all_staff.drop(columns=columns_to_drop, inplace=True)

    # Replace 'Java Script' with 'JavaScript' in the 'familiar_tech' column
    all_staff['familiar_tech'] = all_staff['familiar_tech'].str.replace('Java Script', 'JavaScript')

    # Replace 'JAVA' with 'Java' in the 'familiar_tech' column
    all_staff['familiar_tech'] = all_staff['familiar_tech'].str.replace('JAVA', 'Java')

    # Make sure the type is "string"
    all_staff['familiar_tech'] = all_staff['familiar_tech'].astype(str)

    # Define a function to split each word in a sentence and put them in a list
    def split_tech(tech_string):
        split_list = re.split(r'[;,]', tech_string)
        split_list = [item.strip() for item in split_list if item.strip()]
        
        return split_list
    
    all_staff['familiar_tech'] = all_staff['familiar_tech'].apply(split_tech)

    # Flatten the list of lists into a single list
    flat_list = [lang for sublist in all_staff['familiar_tech'].tolist() for lang in sublist]

    # Use Counter to count the occurrences of each language
    language_count = Counter(flat_list)

    # Select the most common technologies from the applicants
    sorted_counts = sorted(language_count.items(), key=lambda x: x[1], reverse=True)
    threshold = 15
    tech_labels = [lang for lang, count in sorted_counts if count > threshold]

    # Column name mapping
    column_name_mapping = {
        "Start time": "start_time",
        "Completion time": "completion_time",
        "Why do you want to intern at Thaddeus?": "interest_in_Thaddeus",
        "Who referred you to our organization and/or how did you hear about us?": "referred_orig",
        "What type of internship experience are you interested in?": 'internship_type',
        "Which position are you interested in": 'position',
        "Please choose the length of time you are willing to commit to this internship, at this moment.": 'duration',
        "Zip Code": 'zip_code',
        "Do you have any relevant training and/or education you would like to mention?": 'report_education',
        "Name of college/university attended": 'university',
        "Degree or Credentials": 'degree',
        "Year degree/credentials received": 'degree_year',
        "Why do you think you are the perfect fit for this role?": 'interest_in_role'
    }

    all_staff.rename(columns = column_name_mapping, inplace = True)


    # Drop rows with NA values for columns
    all_staff = all_staff.dropna(subset=['Accepted Internship'])


    # Categorical columns: categorize them into numbers or one-hot encoding

    # map column 'Have you already emailed your resume to HR@thaddeus.org?' into 1 and 0
    mapping = {'yes': 1, 'no': 0}
    all_staff['Have you already emailed your resume to HR@thaddeus.org?'] = all_staff['Have you already emailed your resume to HR@thaddeus.org?'].fillna("No").str.strip().str.lower()
    all_staff['Have you already emailed your resume to HR@thaddeus.org?'] = all_staff['Have you already emailed your resume to HR@thaddeus.org?'].map(mapping)

    # map column 'Accepted Internship' into 1 and 0
    mapping = {'true': 1, 'false': 0}
    all_staff['Accepted Internship'] = all_staff['Accepted Internship'].astype(str).str.strip().str.lower()
    all_staff['Accepted Internship'] = all_staff['Accepted Internship'].map(mapping)


    # Encode for "familiar_tech"

    # Function to one-hot encode a row
    def one_hot_encode(row):
        encoding = {label: 1 if label in row else 0 for label in tech_labels}
        return pd.Series(encoding)

    # Apply the one-hot encoding function to each row
    tech_encoding = all_staff['familiar_tech'].apply(one_hot_encode)
    tech_encoding_array = np.array(tech_encoding)


    # #### Encode for "referred_orig"

    def one_hot_encoding(column, labels):
        encoding = pd.get_dummies(column, columns=labels)
        return encoding

    mapping2 = {'Handshake;': 1, 'LinkedIn;': 2, 'Family/Friend;': 3, 'Thaddeus Website;': 4, 'Handshake;Thaddeus Website;': 4, 'Indeed;': 5, 
            'Current/Former Employee/Volunteer;': 6, 'Others;': 7}

    
    all_staff['referred_orig'] = all_staff['referred_orig'].apply(lambda x: 'Others;' if str(x) not in mapping2 else x)

    valu_counts = all_staff['referred_orig'].value_counts()
    custom_labels = ['Handshake', 'LinkedIn', 'Family/Friend', 'Thaddeus Website', 'Handshake;Thaddeus Website;', 'Indeed', 'Current/Former Employee/Volunteer', 'Others']

    source_encoding = one_hot_encoding(all_staff['referred_orig'], custom_labels)
    source_encoding_array = np.array(source_encoding)


    # Generate cosine similarities between column 'position' and other related columns
    df = pd.DataFrame({'position': all_staff['position'], 'Organization/Company Name #1': all_staff['Organization/Company Name #1'], 'Position/Title':all_staff['Position/Title'],
                    'Organization/Company Name #2': all_staff['Organization/Company Name #2'], 'Position/Title2': all_staff['Position/Title2'],
                    'Organization/Company Name #3': all_staff['Organization/Company Name #3'], 'Position/Title3': all_staff['Position/Title3']})

    for column in df.columns:
        df[column] = df[column].str.lower()  # Convert to lowercase
        df[column].fillna('', inplace=True)  # Fill NaN with empty string
        df[column] = df[column].str.replace('[^\w\s]', '', regex=True) # Remove punctuation and stopwords

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(df)
    position_vector = tfidf_vectorizer.transform(df['position'])
    
    # Fit and transform the interested_position and past_company columns 
    for index, column in enumerate(df.columns[1:]):
        tfidf_matrix = tfidf_vectorizer.transform(df[column])
    
        df[f'similarity {index}']  = np.diagonal(cosine_similarity(position_vector, tfidf_matrix))

    # Free response questions: use Word2Vec

    # ### Define Word2Vec model
    def word2vec_average(columns, model):
        vectors = []
        for column in columns:
            if pd.isna(column):
                vectors.append(np.zeros(model.vector_size))
            else:
                word_vectors = [model[word] for word in column.split() if (word in model) and (isinstance(column, str) == True)]
                avg_vectors = np.mean(word_vectors, axis=0)
                vectors.append(avg_vectors)
        return np.array(vectors)


    # ### Incoporate Fastmap algorithm 
    import random

    class FastMap():
        def __init__(self, objects, dist_func):
            self.N = len(objects) 
            self.objects = objects
            self.dist_func = dist_func

        def embed(self, K, max_iters=10, e=0.0001):
            np.random.seed(5)
            self.K = K
            self.P = np.zeros((self.N, self.K))

            for k in range(self.K):
                Oa = np.random.randint(self.N)
                Ob = Oa
                for t in range(max_iters):
                    d_ai = self.single_source_distances(Oa) # an array with lenght N
                    d_ai_new2 = np.power(d_ai, 2) - np.sum(np.power(self.P[Oa, :k] - self.P[:, :k], 2), axis=1)
                    Oc = np.argmax(d_ai_new2)
                    if Oc == Ob:
                        break
                    elif t < max_iters - 1:
                        Ob = Oa
                        Oa = Oc
                        d_ib_new2 = d_ai_new2
            # Fast Pivot ends.
                d_ab_new2 = d_ai_new2[Ob]
                if d_ab_new2 < e: 
                    self.K = k
                    self.P = self.P[:, :k]
                    break
                d_ab_new = np.sqrt(d_ab_new2)
                self.P[:, k] = (d_ai_new2 + d_ab_new2 - d_ib_new2) / (2 * d_ab_new) # k-th coordinates of the embedding
            return self.P
        
        def single_source_distances(self, Os):
            d_si = np.zeros(self.N)
            for i in range(self.N):
                if i != Os:
                    d_si[i] = self.dist_func(self.objects[Os], self.objects[i])
            return d_si	

        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0  # Handle the case of zero vector magnitude
            
            similarity = dot_product / (norm_vec1 * norm_vec2)

            if isinstance(similarity, np.float32):
                return 1 - similarity
            else:
                return 0.0

    # Apply word2vec_model to specific columns and reduce the dimensionalities using FastMap algorithm.
    duty1 = all_staff['What were your duties/responsibilities in this role?']
    duty1_vectors = word2vec_average(duty1, word2vec_model)
    duty1_vectors_fastmap = FastMap(duty1_vectors, FastMap.cosine_similarity)
    duty1_embedding = duty1_vectors_fastmap.embed(K=10)

    duty2 = all_staff['What were your duties/responsibilities in this role?2']
    duty2_vectors = word2vec_average(duty2, word2vec_model)
    duty2_vectors_fastmap = FastMap(duty2_vectors, FastMap.cosine_similarity)
    duty2_embedding = duty2_vectors_fastmap.embed(K=10)

    duty3 = all_staff['What were your duties/responsibilities in this role?3']
    duty3_vectors = word2vec_average(duty3, word2vec_model)
    duty3_vectors_fastmap = FastMap(duty3_vectors, FastMap.cosine_similarity)
    duty3_embedding = duty3_vectors_fastmap.embed(K=10)

    awards = all_staff['Please list any other trainings or awards, honors, special achievements, etc.']
    awards_vectors = word2vec_average(awards, word2vec_model)
    awards_vectors_fastmap = FastMap(awards_vectors, FastMap.cosine_similarity)
    awards_embedding = awards_vectors_fastmap.embed(K=10)

    role_interest = all_staff['interest_in_role']
    role_interest_vectors = word2vec_average(role_interest, word2vec_model)
    role_interest_vectors_fastmap = FastMap(role_interest_vectors, FastMap.cosine_similarity)
    role_interest_embedding = role_interest_vectors_fastmap.embed(K=5)

    relevant_info = all_staff['Please provide any other relevant information (e.g. skills, personal attributes, experience):']
    relevant_info_vectors = word2vec_average(relevant_info, word2vec_model)
    relevant_info_vectors_fastmap = FastMap(relevant_info_vectors, FastMap.cosine_similarity)
    relevant_info_embedding = relevant_info_vectors_fastmap.embed(K=10)

    thaddeus_interest = all_staff['interest_in_Thaddeus']
    thaddeus_interest_vectors = word2vec_average(thaddeus_interest, word2vec_model)
    thaddeus_interest_vectors_fastmap = FastMap(thaddeus_interest_vectors, FastMap.cosine_similarity)
    thaddeus_interest_embedding = thaddeus_interest_vectors_fastmap.embed(K=5)


    # ## Numerical questions: use mediam value for missing values

    # Get median values for each column
    median_val = all_staff.iloc[:, 52: 71].median()

    # Fill in median values for missing values
    all_staff.iloc[:, 52: 71] = all_staff.iloc[:, 52: 71].fillna(median_val)


    # Combine the vectors into a dataframe
    new_staff = pd.DataFrame({'similarity between position applying and past company 1': df['similarity 0'],
                            'similarity between position applying and past position 1': df['similarity 1'], 'similarity between position applying and past company 2': df['similarity 2'],
                            'similarity between position applying and past position 2': df['similarity 3'], 'similarity between position applying and past company 3': df['similarity 4'],
                            'similarity between position applying and past position 3': df['similarity 5']})
    
    # Reset index
    new_staff.reset_index(drop=True, inplace=True)

    # Reset indexes for some columns
    numerical_column = all_staff.iloc[:, 52: 71]
    numerical_column = numerical_column.reset_index(drop=True)
    extra_column = all_staff['Have you already emailed your resume to HR@thaddeus.org?']
    extra_column = extra_column.reset_index(drop=True)
    target_column = all_staff['Accepted Internship']
    target_column = target_column.reset_index(drop=True)

    # Combine new_staff with the numerical columns
    combined_df = pd.concat([new_staff, numerical_column, extra_column, target_column], axis=1)
    combined_df2 = pd.DataFrame.copy(combined_df)

    # Combine combined_df2 with the categorical columns
    temp_list =['duties in role 1', 'duties in role 2', 'duties in role 3', 'Other trainings or awards', 'relevant information', 'interest in role', 'interest in thaddeus']
    for index, i in enumerate([duty1_embedding, duty2_embedding, duty3_embedding, awards_embedding, role_interest_embedding, relevant_info_embedding, thaddeus_interest_embedding]):
        column_name = [temp_list[index]+f' {i}' for i in range(i.shape[1])]
        temp_df = pd.DataFrame(i, columns = column_name)
        combined_df2 = pd.concat([temp_df, combined_df2], axis=1)
    
    temp_df = pd.DataFrame(tech_encoding_array, columns = ['Python','MySQL','JavaScript','Microsoft PowerApps','Java','C++','R','Microsoft PowerAutomate','HTML','CSS','C','Tableau','React'])
    combined_df2 = pd.concat([temp_df, combined_df2], axis=1)

    temp_df = pd.DataFrame(source_encoding_array, columns = ['Handshake', 'LinkedIn', 'Family/Friend', 'Thaddeus Website', 'Handshake;Thaddeus Website;', 'Indeed', 'Current/Former Employee/Volunteer', 'Others'])
    combined_df2 = pd.concat([temp_df, combined_df2], axis=1)

    combined_df2 = combined_df2.astype(float)


    # ## Apply PCA

    # Standardize the data
    scaler = StandardScaler()
    features = combined_df2.drop('Accepted Internship', axis=1)
    scaled_data = scaler.fit_transform(features)

    # Specify the number of components
    n_components = 30
    pca = PCA(n_components=n_components)

    # Fit PCA and transform data
    pca_data = pca.fit_transform(scaled_data)

    # Create a new DataFrame with the retained principal components
    pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i}' for i in range(1, n_components + 1)])

    # Create a DataFrame to store the loadings and associate them with feature names
    pca_loadings_df = pd.DataFrame(pca.components_, columns=features.columns)

    # Transpose the DataFrame to make features the rows and principal components the columns
    pca_loadings_df = pca_loadings_df.transpose()

    # Calculate the absolute values of loadings for each feature and principal component
    pca_loadings_df = pca_loadings_df.abs()

    # Rank the features within each principal component based on their importance (loading magnitude)
    pca_loadings_df['Importance'] = pca_loadings_df.sum(axis=1)
    pca_loadings_df = pca_loadings_df.sort_values(by='Importance', ascending=False)

    # Split the dataset
    X = pca_df
    y = combined_df2['Accepted Internship']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X













