import sys
import joblib
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QComboBox, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator  # Correctly import QDoubleValidator

class PricePredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Price Predictor")
        self.setFixedSize(400, 300)  # Optional: Set a fixed window size

        # Load the saved model and preprocessing objects
        try:
            self.model = joblib.load('linear_regression_model.joblib')
            self.encoders = joblib.load('label_encoders.joblib')
            self.scaler = joblib.load('minmax_scaler.joblib')
            self.feature_names = joblib.load('feature_names.joblib')
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", f"Required file not found: {e}")
            sys.exit(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading models: {e}")
            sys.exit(1)

        # Define categorical and numerical features
        self.categorical_features = [feature for feature in self.feature_names if feature in self.encoders]
        self.numerical_features = [feature for feature in self.feature_names if feature not in self.encoders]

        # Create input widgets
        self.input_widgets = {}
        form_layout = QFormLayout()

        for feature in self.feature_names:
            if feature in self.categorical_features:
                # Categorical feature - use QComboBox
                combo = QComboBox()
                # Populate combo box with original categories (inverse transform)
                original_categories = self.encoders[feature].classes_
                combo.addItems(original_categories)
                self.input_widgets[feature] = combo
                form_layout.addRow(QLabel(feature), combo)
            else:
                # Numerical feature - use QLineEdit with QDoubleValidator
                line_edit = QLineEdit()
                line_edit.setPlaceholderText(f"Enter {feature}")
                validator = QDoubleValidator()
                validator.setBottom(0)  # Optionally set the minimum value
                line_edit.setValidator(validator)
                self.input_widgets[feature] = line_edit
                form_layout.addRow(QLabel(feature), line_edit)

        # Predict button
        self.predict_button = QPushButton("Predict Price")
        self.predict_button.clicked.connect(self.predict_price)

        # Result label
        self.result_label = QLabel("Predicted Price: ")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.predict_button)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def predict_price(self):
        try:
            input_data = {}
            for feature, widget in self.input_widgets.items():
                if feature in self.categorical_features:
                    # Categorical feature - get the current text and encode it
                    value = widget.currentText()
                    encoded = self.encoders[feature].transform([value])[0]
                    input_data[feature] = encoded
                else:
                    # Numerical feature - get the float value
                    value_str = widget.text()
                    if value_str == '':
                        raise ValueError(f"Please enter a value for {feature}.")
                    value = float(value_str)
                    input_data[feature] = value

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Normalize numerical features
            input_df[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])

            # Ensure the order of columns matches the training data
            input_df = input_df[self.feature_names]

            # Predict
            predicted_price = self.model.predict(input_df)[0]

            # Display the result
            self.result_label.setText(f"Predicted Price: {predicted_price:.2f}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    window = PricePredictorApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
