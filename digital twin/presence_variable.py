import pandas as pd

class PresenceVariable:
    def __init__(self, allowed_moe, reference_data):
        self.margin_error = allowed_moe
        self.reference_data = reference_data

    def transform_data(self, input_data, result_fields):
        result = inputdata[result_fields]
        return result

    def calculate_distribution(self):
        pass