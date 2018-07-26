import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_files(df):
    """
    Traverses through each row of the dataframe and reads
    files from locations mentioned in 2 columns(1 having
    locations of content files and the other having locations
    of category files) of the dataframe
    and puts them to two different lists.
    Parameters
    -----------
    df : pandas dataframe having 2 columns named as 'file_loc', 'cat_file_loc'
    Returns
    -----------
    content_list : List of the contents of all the email content files
    cat_list : List of the contents of all category files
    Note: The lists have 1 to 1 correspondence.
    """
    content_list = []
    cat_list = []
    for i, row in df.iterrows():
        if i == 0:
            content_list.append('')
            cat_list.append('')
            continue
        with open(row['file_loc']) as f:
            content = f.read()
            content_list.append(content)
        with open(row['cat_file_loc']) as f:
            cat_content = f.read()
            cat_list.append(cat_content)
    return content_list, cat_list


def get_cat(x, num):
    """
    Given a string like '1,1,2\n3,7,2\n4,1,2' and a number the
    function extracts the second-level category
    For a substring n1,n2,n3
    n1 = top-level category
    n2 = second-level category
    n3 = frequency with which this category was assigned to this message
    Parameters
    -----------
    x : a string containing 3 integered subsequences separated
    by new-line character as shown above
    num : 1st integer in a substring
    It can take values 1, 2, 3, 4
    1 Coarse genre
    2 Included/forwarded information
    3 Primary topics (if coarse genre 1.1 is selected)
    4 Emotional tone (if not neutral) 
    Returns
    -----------
    category id : If num is present as the 1st integer in any of
    the sbusequence then it is the 2nd level category else it is None.
    """
    cats = x.strip('\n').split('\n')
    for cat in cats:
        if cat.startswith(str(num)):
            split_val = cat.split(',')
            if len(split_val) > 1:
                return split_val[1]
            else:
                return None


def parse_raw_message(raw_message):
    """
    Extracts useful information like to, from, subject, body
    from a text blob of email conversation
    Parameters
    -----------
    raw_message : raw email content
    Returns
    -----------
    email : A dictionary object containing to, from, subject, body
    as keys and corresponding entries as values
    """
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to', 'subject']
    for line in lines:
        if ':' not in line:
            message += line.strip() + ' '
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email


def map_to_list(emails, key):
    """
    Given a list of dictionary objects returns a list containing the values
    corresponding to a particular key in each of the dictionary
    Parameters
    -----------
    emails : list of dictionaries
    key : key for which values need to be extracted
    Returns
    -----------
    results : list containing all the values corresponding
    to the key in all the dictionaries
    """
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


def parse_into_emails(messages):
    """
    Given a list of raw contents from emails it generates a single
    dictionary containing values as lists
    Parameters
    -----------
    messages : list of raw email content
    Returns
    -----------
    dictionary : subject, body, to, from as keys and each of them have values
    in the form of lists with length same as that of messages
    """
    emails = [parse_raw_message(message) for message in messages]
    return {
        'subject': map_to_list(emails, 'subject'),
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }


def top_feats_in_doc(X, features, row_id, top_n=25):
    """
    For a particular document it generates a data frame containing
    top n features and their TF-IDF scores.
    -----------
    X : Sparse matrix generated using TF-IDF vectorizer
    features : Array containing the feature name
    row_id : row id corresponds to a particular document
    top_n : top n features to extract, default 25
    Returns
    -----------
    data frame : pandas data frame containing the top_n features and their scores
    for a particular document
    """
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """
    Generates a data frame containing top n features and their TF-IDF scores
    across all the documents.
    -----------
    X : Sparse matrix generated using TF-IDF vectorizer
    features : Array containing the feature name
    grp_ids : Indices of documents to be considered for generating the top features
    min_tfidf : default 0.1
    top_n : top n features to extract, default 25
    Returns
    -----------
    data frame : pandas data frame containing the top_n features and their scores
    across all the documents
    """
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_tfidf_feats(row, features, top_n=20):
    """
    Given two arrays of features and corresponding scores it creates a
    dat frame by taking top n features into account.
    -----------
    row : An array of TF-IDF scores corresponding to each feature
    features : Array containing the feature name
    top_n : top n features to extract, default 20
    Returns
    -----------
    df : pandas data frame containing the top_n features and their scores
    """
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
  