# Data for Accuracy, Precision, Recall, and F1 Score across different algorithms
accuracy_data = [0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086, 0.8641304347826086]
precision_data = [0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887, 0.8867924528301887]
recall_data = [0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962, 0.8785046728971962]
f1_score_data = [0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206, 0.8826291079812206]

# Calculate the average for each metric
average_accuracy = sum(accuracy_data) / len(accuracy_data)
average_precision = sum(precision_data) / len(precision_data)
average_recall = sum(recall_data) / len(recall_data)
average_f1_score = sum(f1_score_data) / len(f1_score_data)

print("Average Accuracy:", average_accuracy)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1_score)