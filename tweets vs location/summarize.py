"""
sumarize.py
"""
import pickle
import numpy as np

def main():
    cluster = pickle.load(open('cluster_result.pkl', 'rb'))
    classify = pickle.load(open('classify_result.pkl', 'rb'))
    with open("summarize.txt", "w") as text_file:
        text_file.write("Summary")
        text_file.write("\nNumber of users collected: %d"%cluster['num_users'])
        text_file.write("\nNumber of messages collected: %d"%(classify['num_training']+classify['num_testing']))
        text_file.write(" (training message: %d, testing messages: %d)"%(classify['num_training'],classify['num_testing']))
        text_file.write("\nNumber of communities discovered: %d"%len(cluster['community_sizes']))
        text_file.write("\nAverage number of users per community: %0.3f"%np.mean(cluster['community_sizes']))
        text_file.write("\nNumber of instances per class found: ")
        text_file.write(str(classify['stats']))
        text_file.write("\nTop misclassified messages - ")
        for i in classify['misclassified'].keys():
            text_file.write("\npredict %d, true %d, error %0.5f"%(int(classify['misclassified'][i]['predict']),int(classify['misclassified'][i]['true']),classify['misclassified'][i]['error'] ))
            text_file.write("\ntweet text: %s"%classify['misclassified'][i]['text'])
        text_file.write("\nTop correctly classified messages - ")
        for i in classify['correct_classified'].keys():
            text_file.write("\npredict %d, true %d, error %0.5f"%(int(classify['correct_classified'][i]['predict']),int(classify['correct_classified'][i]['true']),classify['correct_classified'][i]['error'] ))
            text_file.write("\ntweet text: %s"%classify['correct_classified'][i]['text'])

if __name__ == '__main__':
    main()
