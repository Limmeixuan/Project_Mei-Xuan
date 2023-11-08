# data for Accuracy, Precision, Recall, and F1 Score across different algorithms
accuracy_data = [0.8858695652173914 , 0.8967391304347826 , 0.8695652173913043 , 0.8967391304347826 , 0.8858695652173914, 0.875, 0.8858695652173914, 0.8641304347826086, 0.8804347826086957, 0.8913043478260869]
precision_data = [0.8981481481481481, 0.9150943396226415, 0.9029126213592233, 0.9150943396226415, 0.9134615384615384, 0.8962264150943396, 0.9134615384615384, 0.9019607843137255, 0.9047619047619048, 0.9142857142857143]
recall_data = [0.9065420560747663, 0.9065420560747663, 0.8691588785046729, 0.9065420560747663, 0.8878504672897196, 0.8878504672897196, 0.8878504672897196, 0.8598130841121495, 0.8878504672897196, 0.897196261682243]
f1_score_data = [0.9023255813953489, 0.9107981220657277, 0.8857142857142858, 0.9107981220657277, 0.9004739336492891, 0.892018779342723, 0.9004739336492891, 0.8803827751196173, 0.8962264150943396, 0.9056603773584906]

# Calculate the average
average_accuracy = sum(accuracy_data) / len(accuracy_data)
average_precision = sum(precision_data) / len(precision_data)
average_recall = sum(recall_data) / len(recall_data)
average_f1_score = sum(f1_score_data) / len(f1_score_data)

print("Average Accuracy:", average_accuracy)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1 Score:", average_f1_score)