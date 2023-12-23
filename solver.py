

# Set your OpenAI API key
# openai.api_key = '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'



""

import os
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import textwrap
from scipy.sparse.csgraph import laplacian

# MODEL="gpt-4-1106-preview"
MODEL="gpt-3.5-turbo-1106"
API_KEY="your-key"
def isfloat(str):
    try: 
        float(str)
    except ValueError: 
        return False
    return True

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        self.client = OpenAI(api_key=API_KEY)
        self.messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
        ]
    
    def calc_similarity(self,a,b):
        messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
            {"role": "user", "content": f"On a scale from 0 to 10, what is the logical relationship between the following two sentences? 0 represents being totally contradicting, 10 represents being totally the same, and 5 represents they are independent. your answer should only return a real number without any other text. your answer should only return a real number without any other text.\nsentence 1:{a};\nsentence 2:{b}"}
        ]
        response=self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            seed=12,
        )
        result=response.choices[0].message.content
        messages.append({"role": "assistant", "content": result})
        count=0
        while not isfloat(result):
            count+=1
            if count>5:
                return 0
                # raise ValueError
            self.messages.append({"role": "user", "content": "give a numerical answer only"})
            response=self.client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            seed=12
            )
            result=response.choices[0].message.content

        return response.choices[0].message.content
    
    def aggregate_thoughts(self,problem,thoughts):
        messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
            {"role": "user", "content": f"You are trying to solve this problem: {problem}\nHere is a subset of belief you generated:{';'.join(thoughts)}. Now you aggregate these thoughts into a single thoughts, note that do not attempt to solve this problem yet, just aggregate the thoughts, make sure you include the details. Return your answer directly."}
        ]
        response=self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            seed=12
        )
        return response.choices[0].message.content


    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response=self.client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            seed=12
        )
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    
    def chat_onetime(self,message):
        messages=[
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure"},
        ]
        messages.append({"role": "user", "content": message})
        response=self.client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            seed=12
        )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    def reset(self):
        self.messages = [
            {"role": "system", "content": "You are an assistant operating under a multi-stage thinking structure. "},
        ]
# bot=ChatGPT()
# while 1:
#     print(bot.chat(input()))


class solver:
    def __init__(self):
        self.bot1=ChatGPT()
        self.bot2=ChatGPT()
        self.questions_understanding=[
    "How many and which specific elements or objects are involved in this problem?",
    "What are the intrinsic properties and characteristics of these elements or objects?",
    "What is the relationship or interaction between these elements or objects?"]


    def tot_prompt(self,problem,thoughts):
        return f"""
        Imagine three different experts are answering this question.
        All experts will write down 1 step of their thinking,
        then share it with the group.
        Then all experts will go on to the next step, etc.
        If any expert realises they're wrong at any point then they leave.
        The question {problem}, Some fact you might find useful include: {thoughts}
        """
        # https://github.com/dave1010/tree-of-thought-prompting
    
    def cot_prompt(self,problem,thoughts):
        return f"The question {problem}, Some fact you might find useful include: {thoughts}. Now lets plan to solve this problem step by step. After that, revise your solution, and give me the final result."


    def similarity_score(self,thoughts1,thoughts2):
        response=self.bot2.calc_similarity(thoughts1,thoughts2)
        score=np.float32(response)
        if score<0:
            score=np.float32(0)
        return score

    def generate_matrix(self, thoughts):
        n = len(thoughts)
        similarity_matrix = np.zeros((n, n))

        # Compute similarity scores
        for i in range(n):
            for j in range(i+1, n):  # Skip diagonal and redundant computations
                score = self.similarity_score(thoughts[i], thoughts[j])
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score

   
        np.fill_diagonal(similarity_matrix, 0)

        # Normalize the matrix
        # min_score = np.min(similarity_matrix[np.nonzero(similarity_matrix)])
        # max_score = np.max(similarity_matrix)

        # # Avoid division by zero in case all values are the same
        # if max_score != min_score:
        #     similarity_matrix = (similarity_matrix - min_score) / (max_score - min_score)

        return similarity_matrix
    
    def aggregate(self,problem,thoughts,labels):
        unique_labels = np.unique(labels)
        
        aggregated_thoughts = []

        for label in unique_labels:
            # Find the subset of thoughts corresponding to the current label
            subset_thoughts = [thought for thought, lbl in zip(thoughts, labels) if lbl == label]

            # Call the aggregate_thoughts method and store the result
            aggregated_thought = self.bot2.aggregate_thoughts(problem, subset_thoughts)
            aggregated_thoughts.append(aggregated_thought)

        # Print the number of unique labels
        # print(f"Number of labels: {len(unique_labels)}")

        return aggregated_thoughts
   
    def spectral_clustering_auto_select(self,matrix):
        # Compute the Laplacian
        L, _ = laplacian(matrix, normed=True, return_diag=True)
        eigenvalues = np.linalg.eigvalsh(L)

        eigenvalues.sort()

        gaps = np.diff(eigenvalues)
        k = np.argmax(gaps) + 1
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
        labels = clustering.fit_predict(matrix)

        return labels, k

    def thought_clustering(self,thoughts):
        mat=self.generate_matrix(thoughts)
        n_clusters=3
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = clustering.fit_predict(mat)
        # labels, n_clusters=self.spectral_clustering_auto_select(mat)

        new_mat = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    indices_i = np.where(labels == i)[0]
                    indices_j = np.where(labels == j)[0]
                    sum_weights = np.sum(mat[np.ix_(indices_i, indices_j)])
                    n_edges = len(indices_i) * len(indices_j)
                    new_mat[i, j] = sum_weights / n_edges if n_edges > 0 else 0
        

        return mat,labels,new_mat
        

    def visualize_graph(self, name, thoughts, matrix, labels):
        G = nx.Graph()

        for i, thought in enumerate(thoughts):
            G.add_node(i, label=textwrap.fill(thought, width=45))

        # Add edges with weights
        for i in range(len(thoughts)):
            for j in range(len(thoughts)):
                if i != j and matrix[i][j] != 0:  # Assuming a weight of 0 means no edge
                    G.add_edge(i, j, weight=matrix[i][j])

        # Set up the layout
        pos = nx.spring_layout(G)

        # Set a larger figure size
        plt.figure(figsize=(22, 18))

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, cmap=plt.cm.RdYlBu, node_color=labels)

        # Draw the edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        # Draw the labels with a smaller font size
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)

        # Format edge weights to three decimal points and draw them
        edge_labels = {edge: f"{weight:.3f}" for edge, weight in nx.get_edge_attributes(G, 'weight').items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Save the figure
        plt.savefig(f"{name}.png")
        plt.close()
        # Show the plot
        # plt.show()


    def understand(self, problem, num_thoughts):
        filtered_thoughts = []

        for question in self.questions_understanding:
            # Ask each question individually
            response = self.bot1.chat(f"The problem statement you are trying to solve is: '{problem}'. Please answer the following question: {question}return the answers only. Return exacctly {num_thoughts//3} bullet points for this question. your answer should be in full sentence. Your answer should be separated by '\n'" )
            # Extract the answer
            thoughts = response.split("\n")
            # Filter out short responses
            valid_thoughts = [element for element in thoughts if len(element) >= 2]

            # Add valid thoughts to the final list
            filtered_thoughts.extend(valid_thoughts)

            # Break the loop if the number of desired thoughts is reached
            if len(filtered_thoughts) >= num_thoughts:
                break

        # Return the first 'num_thoughts' thoughts
        return filtered_thoughts[:num_thoughts]


    # def understand(self,problem,num_thoughts):
    #     response=self.bot1.chat(f"The problem statement you are trying to solve is {problem}, before you make any attempt to solve it, firstly decompose the problem by answering these {len(self.questions_understanding)} questions:{''.join(self.questions_understanding)}.return the answers only. Return exacctly {num_thoughts} bullet points for these 3 questions in total. your answer should be in full sentence. Your answer should be separated by \\n \n example:\n what can you say about today's weather? \n Answer: It is sunny\nIt is windy")
    #     thoughts=response.split("\n")
    #     filtered_thoughts = [element for element in thoughts if len(element) >= 2]

    #     print(filtered_thoughts)

    #     return filtered_thoughts

    def run(self, problem):
        try:
            self.bot1.reset()
            self.bot2.reset()
            thoughts = self.understand(problem, 15)
            # print(thoughts)
            mat, label, mat_agg = self.thought_clustering(thoughts)
            # self.visualize_graph("before_agg", thoughts, mat, label)
            aggregated_thoughts = self.aggregate(problem, thoughts, label)
            # print()
            # print(aggregated_thoughts)
            # self.visualize_graph("after_agg", aggregated_thoughts, mat_agg, np.unique(label))
            self.bot2.reset()
            self.bot2.chat(self.cot_prompt(problem, aggregated_thoughts))
            # print(self.bot2.messages)
            return self.bot2.chat("now return the numerical part of the answer only, do not return anything other then numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return 0
    
    def run_no_agg(self, problem):
        try:
            self.bot1.reset()
            self.bot2.reset()
            thoughts = self.understand(problem, 9)
            self.bot2.reset()
            self.bot2.chat(self.cot_prompt(problem, thoughts))
            # print(self.bot2.messages)
            return self.bot2.chat("now return the numerical part of the answer only, do not return anything other then numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return 0

    def run_base(self,problem):
        try:
            self.bot1.reset()
            self.bot2.reset()
            self.bot1.chat(problem)
            return self.bot1.chat("now return the numerical part of the answer only, do not return anything other then numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "0"
        
        

    


def main():
    solve=solver()
    answer=solve.run("Bill is ordering a new truck. He has decided to purchase a two-ton truck with several added features: a king cab upgrade, a towing package, leather seats, running boards, and the upgraded exterior light package. The base price of the truck is $30,000, and the other features are at extra cost. The king cab is an extra $7,500, leather seats are one-third the cost of the king cab upgrade, running boards are $500 less than the leather seats, and the upgraded exterior light package is $1500. What is the total cost of Bill's new truck, in dollars?")
    # answer=solve.run("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
    print(answer)

if __name__ == "__main__":
    main()

