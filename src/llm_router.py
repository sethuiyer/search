from txtai.pipeline import Labels

class InstructionClassifier:
    def __init__(self):
        # Initialize the labels model
        self.labels = Labels('facebook/bart-large-mnli')

        self.tags = [
            "Programming",
            "Factual",
            "Creative Writing",
            "Roleplaying"
        ]

        self.tools_labels = ["Real Time Information needed: Available in Internet",
                             "Historic Information needed: Available in Wikipedia",
                             "Sufficient Information"]


    def classify_instructions(self, data):
        result = []
        for text in data:
            # Predict tags
            tag_labels_result = self.labels(text, self.tags)
            tag_label = self.tags[tag_labels_result[0][0]] if tag_labels_result[0][0] < len(self.tags) else "Unknown"

            tool_labels_result = self.labels(text, self.tools_labels)
            tool_label = self.tools_labels[tool_labels_result[0][0]] if tool_labels_result[0][0] < len(self.tools_labels) else "Unknown"

            result.append((text, tag_label, tool_label))
        return result
