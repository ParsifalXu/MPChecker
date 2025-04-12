import os 
import json
from openai import OpenAI

client = OpenAI(api_key="YOUR_OWN_API") 

from tools.macros import INFO_DIR
from tools.macros import RES_DIR



def process_document(project):
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    info_path = os.path.join(INFO_DIR, project)
    if not os.path.exists(info_path):
        print(f"{project} is not exist, please download and extract info first")
        exit(0)

    res_path = os.path.join(RES_DIR, project)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    def read_doc_and_params(dict):
        docstring = ""
        if dict['param']:
            docstring += "Parameters\n----------"
        for param in dict['param']:
            docstring +=  "\n" + param + ":"
            docstring += dict['param'][param]

        if dict['attr']:
            docstring += "\nAttributes\n----------"
        for attr in dict['attr']:
            docstring +=  "\n" + attr + ":"
            docstring += dict['attr'][attr]

        param_list = ','.join(dict['pa'].keys())
        return docstring, param_list


    folders = os.listdir(info_path)
    for folder in folders:
        if "DS_Store" in folder:
            continue
        if os.path.exists(f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"):
            print(f"Calculating constraints ----- {folder}")
            pa_path = f"{INFO_DIR}/{project}/{folder}/{folder}_pa.json"
            with open(pa_path, 'r') as fpa:
                pa = json.load(fpa)
            fpa.close()
            docstring, param_list = read_doc_and_params(pa)
            promptA = "From now on, act as a code developer who is reading the code documentation. Please remember this paramter list mentioned in this documentation:[" + param_list + "].Please extract all parameter information with their types and default values from following documents."
            if len(docstring) > 1000:
                messages = construct_messages(docstring, promptA, folder)
            else:
                messages=[
                {
                    "role": "system",
                    "content": "You are a senior python code developer."
                },
                {
                    "role": "user",
                    "content": promptA
                },

                {
                    "role": "user",
                    "content": docstring
                },
                {
                    "role": "user",
                    "content": "Based on the results, pay close attention to the constraints, and give me ALL the constraints details between parameters. Let's think step by step."
                },
                {
                    "role": "user",
                    "content": "Your final task is to convert textual constraints from documentation into a specified logic format. Instructions: 1. Logic Symbols: Use `->` to denote implication (if...then); Use `!` for negation (NOT); Use `^` for logical AND; Use `||` for logical OR; Enclose expressions in parentheses `()` to clarify the order of operations. 2. Keyword Placeholder Usage: If a constraint contains any of the following keywords: \"override\", \"specify\", \"have an effect\”,\”no effect\”, \”significant\", \"ignore\", use these keywords as placeholders within your logic expression. 3. If a parameter is equal to neither 'None' nor any other value, only expressions with other values are retained. 4. Solution Format: Present your solutions as follows: Constraint Number: Start with a sequential number; Text Constraint: Copy the constraint directly from the documentation; Logical Format: Provide the corresponding logic format based on the text constraint. Example: ```1. Text Constraint: \"n_clusters must be None if distance_threshold is not None.\" Logical Format: (!(distance_threshold = 'None')) -> (n_clusters = 'None'). ``` 4. Examples for Clarification: Provide clear examples to illustrate the conversion from text constraint to logic format. This includes handling complex conditions and using keywords as placeholders. 5. Final Verification: After conversion, verify if the parameters used in the logic format match actual parameters from the documentation. If a parameter does not exist, exclude the corresponding constraint from your results. Examples: 1. Text Constraint: \"gamma is only significant for 'rbf', 'poly', and 'sigmoid' kernels.\" Logical Format: ((kernel = 'rbf' || kernel = 'poly') || kernel = 'sigmoid') -> (gamma = 'significant'). 2. Text Constraint: \"n_init: int, default=10. Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. Only used if assign_labels='kmeans'.\" Logical Format: (assign_labels = 'kmeans') -> !(n_init = 'None'). 3. Text Constraint: \"gamma:float, default=1.0 Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for affinity='nearest neighbors'.\" Logical Format: (affinity = 'nearest_ neighbors') -> (gamma = ‘ignore’). Remember: The goal is to accurately translate the provided constraints into the defined logic format, ensuring all expressions are correct and reflect the documentation's intent."
                    }
                ]

            response_content = stream_gpt_response(messages)
            with open(f"{RES_DIR}/{project}/{folder}.txt", 'w') as file:
                file.write(response_content)


def stream_gpt_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages = prompt,
        temperature=0
    )
    return stream.choices[0].message.content



def construct_messages(docstring, promptA, key, chunk_size=1500):
    def split_into_chunks(text, chunk_size):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_into_chunks(docstring, chunk_size)
    total_chunks = len(chunks)

    messages = [
        {
            "role": "system",
            "content": "You are a senior python code developer."
        },
        {
            "role": "user",
            "content": promptA
            }
    ]

    for i, chunk in enumerate(chunks, start=1):
        start_message = f"[START CHUNK {i}/{total_chunks}]\n### {key} ###\n"
        end_message = f"\n[END CHUNK {i}/{total_chunks}]"
        chunk_message = {
            "role": "user",
            "content": start_message + chunk + end_message
        }
        messages.append(chunk_message)

    prompt2 = {
                "role": "user",
                "content": "Based on the results, pay close attention to the constraints, and give me ALL the constraints details between parameters. Let's think step by step."
                }
    prompt3 = {
                "role": "user",
                "content": "Your final task is to convert textual constraints from documentation into a specified logic format. Instructions: 1. Logic Symbols: Use `->` to denote implication (if...then); Use `!` for negation (NOT); Use `^` for logical AND; Use `||` for logical OR; Enclose expressions in parentheses `()` to clarify the order of operations. 2. Keyword Placeholder Usage: If a constraint contains any of the following keywords: \"override\", \"specify\", \"have an effect\”,\”no effect\”, \”significant\", \"ignore\", use these keywords as placeholders within your logic expression. 3. Solution Format: Present your solutions as follows: Constraint Number: Start with a sequential number; Text Constraint: Copy the constraint directly from the documentation; Logical Format: Provide the corresponding logic format based on the text constraint. Example: ```1. Text Constraint: \"n_clusters must be None if distance_threshold is not None.\" Logical Format: (!(distance_threshold = 'None')) -> (n_clusters = 'None'). ``` 4. Examples for Clarification: Provide clear examples to illustrate the conversion from text constraint to logic format. This includes handling complex conditions and using keywords as placeholders. 5. Final Verification: After conversion, verify if the parameters used in the logic format match actual parameters from the documentation. If a parameter does not exist, exclude the corresponding constraint from your results. Examples: 1. Text Constraint: \"gamma is only significant for 'rbf', 'poly', and 'sigmoid' kernels.\" Logical Format: ((kernel = 'rbf' || kernel = 'poly') || kernel = 'sigmoid') -> (gamma = 'significant'). 2. Text Constraint: \"n_init: int, default=10. Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. Only used if assign_labels='kmeans'.\" Logical Format: (assign_labels = 'kmeans') -> !(n_init = 'None'). 3. Text Constraint: \"gamma:float, default=1.0 Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for affinity='nearest neighbors'.\" Logical Format: (affinity = 'nearest_ neighbors') -> (gamma = ‘ignore’). Remember: The goal is to accurately translate the provided constraints into the defined logic format, ensuring all expressions are correct and reflect the documentation's intent."                }

    messages.append(prompt2)
    messages.append(prompt3)

    return messages

