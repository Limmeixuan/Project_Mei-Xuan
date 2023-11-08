# Data for Accuracy, Precision, Recall, and F1 Score across different algorithms
accuracy_data = [0.842391304347826, 0.8260869565217391, 0.8315217391304348, 0.8206521739130435, 0.8206521739130435, 0.8315217391304348, 0.842391304347826, 0.8260869565217391, 0.8369565217391305, 0.8260869565217391]
precision_data = [0.9148936170212766, 0.9120879120879121, 0.9130434782608695, 0.9111111111111111, 0.9111111111111111, 0.9222222222222223, 0.9148936170212766, 0.9213483146067416, 0.9139784946236559, 0.9120879120879121]
recall_data = [0.8037383177570093, 0.7757009345794392, 0.7850467289719626, 0.7663551401869159, 0.7663551401869159, 0.7757009345794392, 0.8037383177570093, 0.7663551401869159, 0.794392523364486, 0.7757009345794392]
f1_score_data = [0.8557213930348259, 0.8383838383838383, 0.8442211055276382, 0.8324873096446701, 0.8324873096446701, 0.8426395939086295, 0.8557213930348259, 0.8367346938775511, 0.8500000000000001, 0.8383838383838383]

# Calculate the average for each metric
average_accuracy = sum(accuracy_data) / len(accuracy_data)
average_precision = sum(precision_data) / len(precision_data)
average_recall = sum(recall_data) / len(recall_data)
average_f1_score = sum(f1_score_data) / len(f1_score_data)

print("Average Accuracy:", average_accuracy)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1_score)