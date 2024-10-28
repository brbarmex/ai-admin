import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

path_directory: str = "./chroma_data"
files_directory: str = "./files"
collection_name: str = "sales"

def load_data(collection: chromadb.Collection):
    for file in os.listdir(files_directory):
        if file.endswith('.txt'):
            with open(os.path.join(files_directory, file), 'r') as f:
                
                r = collection.get(
                    ids=[file],
                    where={"source": file},
                )
                
                if len(r['ids']) > 0:
                    continue
                
                collection.add(
                    ids=[file],
                    metadatas=[{"source":file}],
                    documents=[f.read()]
                ) 
 
def create_prompt(query, resumes):
    resumes = "\n".join([f"<resume>{r}</resume>" for r in resumes])
    return f"""
    <resumes>
    {resumes}
    </resumes>

    <question>
    {query}
    </question>

    Com base nos seguintes dados de vendas Quero um resumo das vendas e recomendações para melhorar as vendas.<resumes>.
    """

def query_groq(prompt) -> str:
    api_key=os.environ['GROQ_API_KEY']
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

def main():
    chromadb_client = chromadb.PersistentClient(path=path_directory)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = chromadb_client.get_or_create_collection(name=collection_name,embedding_function=embedding_function)
    
    load_data(collection=collection)
    
    while True:
        query = input("\nConsulta: ")
        if query.lower() == 'sair':
            break
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=['documents']
        )
        
        prompt = create_prompt(query, results['documents'][0])
        answer = query_groq(prompt)
        
        print(f"Resultados: {answer}")
    
if __name__ == "__main__":
        main()