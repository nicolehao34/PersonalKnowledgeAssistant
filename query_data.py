import argparse
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    # create CLI (command line interface) for the app
    # This allows the user to run the app from the commandpip  line and pass in arguments
    # The argparse module is used to parse command line arguments  
    parser = argparse.ArgumentParser()
    
    # add an argument for the query text
    # This argument is required and must be a string
    # The help parameter provides a description of the argument (help = "The query text.")
    parser.add_argument("query_text",type = str, help="The query text.")
    
    # add an argument for the number of results to return
    # This argument is optional and defaults to 5 if not provided
    # The type parameter specifies the type of the argument (type = int)
    args = parser.parse_args()
    
    query_text = args.query_text
    
    # prepare the db
    
    db = Chroma(
        persist_directory="crhoma_store",
        embedding_function=OpenAIEmbeddings(open_api_key= ""),
    )
    
    # Search the DB
    # The similarity search method is used to find the most similar documents to the query text
    # The k parameter specifies the number of results to return
    # here, retrieving the top 3 most similar documents
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # results is a list of tuples, where each tuple contains a document and its relevance score
    # each tuple contains a document and its relevance score
    
    # Print the results if any are found
    if len(results) == 0 or results[0][1] < 0.7: 
        print(f"Unable to find matching results.")
        return
    
    context_text = "\n\n--- \n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    
if __name__ == "__main__":
    main()