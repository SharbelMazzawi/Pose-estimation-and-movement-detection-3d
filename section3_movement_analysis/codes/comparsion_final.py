import pandas as pd
import os
import json

class CSVComparator:
    def __init__(self, file1='ground_truth.csv', file2='results.csv', output_file='mismatched_rows.xlsx'):
        """
        Initializes the CSVComparator with two file paths.

        Args:
            file1 (str): Path to the first CSV file (ground truth).
            file2 (str): Path to the second CSV file (results).
            output_file (str): Path to save the mismatched rows.
        """

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")

        with open(config_path) as f:
            config = json.load(f)
            self.config = config['comparison']

        self.ground_truth = os.path.join(os.getcwd(), self.config["ground_truth_path"])
        self.results = os.path.join(os.getcwd(), "output_files", self.config["results_path"])
        self.output_file = os.path.join(os.getcwd(), output_file)

    def load_and_clean_csv(self, file_path):
        """ Reads and cleans CSV data. """
        try:
            df = pd.read_csv(file_path, header=0)

            df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
            df = df.dropna().reset_index(drop=True)

            return df

        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")

    def compare_filtered_rows(self):
        """ Compares filtered rows and saves mismatched ones. """
        try:
            df_ground_truth = self.load_and_clean_csv(self.ground_truth)
            df_results = self.load_and_clean_csv(self.results)

            df_ground_truth_filtered = df_ground_truth[df_ground_truth.iloc[:, 4] != 0]  # filter (we get the rows with movement)  
            print(f"\nTotal frames of movement: {len(df_ground_truth_filtered)}")
            selected_frames = set(df_ground_truth_filtered.iloc[:, 0])  # get the frames with movement (no duplicates)
            df_results_filtered = df_results[df_results.iloc[:, 0].isin(selected_frames)] # get the frames from results

            missing_frames = selected_frames - set(df_results.iloc[:, 0])
            print(f"DEBUG: Frames in ground_truth but missing in results: {missing_frames}")


            if df_ground_truth_filtered.empty or df_results_filtered.empty:
                print("No matching frames found.")
                return 0.0

            # # to sort the frames by number
            # df_ground_truth_filtered = df_ground_truth_filtered.sort_values(by=df_ground_truth_filtered.columns[0])
            # df_results_filtered = df_results_filtered.sort_values(by=df_results_filtered.columns[0])

            data1 = df_ground_truth_filtered.iloc[:, 1:4].to_numpy()
            data2 = df_results_filtered.iloc[:, 1:4].to_numpy()

            min_rows = min(len(data1), len(data2))
            data1, data2 = data1[:min_rows], data2[:min_rows]

            mismatched_indices = ~((data1 == data2).all(axis=1)) # get the indices of the mismatched rows
            mismatched_indices = mismatched_indices.nonzero()[0] # list of the indices of the mismatched rows

            mismatched_rows1 = df_ground_truth_filtered.iloc[mismatched_indices, :4]
            mismatched_rows2 = df_results_filtered.iloc[mismatched_indices, :4]

            mismatched_rows1.reset_index(drop=True, inplace=True)
            mismatched_rows2.reset_index(drop=True, inplace=True)

            mismatched_df = pd.concat([mismatched_rows1, mismatched_rows2], axis=1)
            mismatched_df.columns = [f'ground_truth{i+1}' for i in range(4)] + [f'results{i+1}' for i in range(4)]

            mismatched_df.to_excel(self.output_file, index=False, sheet_name="Mismatched Rows")
            print(f"\nMismatched rows saved to {self.output_file}")

            matching_rows = (len(df_ground_truth_filtered) - len(mismatched_indices))
            similarity_percentage = (matching_rows / min_rows) * 100 if min_rows > 0 else 0.0
            print(f"\nExact Row Similarity (all frames): {similarity_percentage:.2f}%")

            self.extract_sequences_to_new_sheet(self.output_file, min_rows)
            
            return similarity_percentage

        except Exception as e:
            raise ValueError(f"Error during comparison: {e}")

    def extract_sequences_to_new_sheet(self, excel_file, min_rows, sequence_threshold=8):
        """
        Extracts long sequences of mismatched rows and calculates their percentage.
        """
        try:
            df = pd.read_excel(excel_file, sheet_name="Mismatched Rows")

            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df = df.dropna(subset=[df.columns[0]])  # ensure numeric values only
            df.iloc[:, 0] = df.iloc[:, 0].astype(int)

            numbers = df.iloc[:, 0].tolist()
            sequences = []
            temp_sequence = []

            for i in range(len(numbers) - 1):
                if numbers[i + 1] == numbers[i] + 1:
                    temp_sequence.append(df.index[i])  # store actual row index
                else:
                    if len(temp_sequence) > sequence_threshold:
                        sequences.extend(temp_sequence)
                    temp_sequence = []

            if len(temp_sequence) > sequence_threshold:  #to add the last sequence (where the for loop ends)
                sequences.extend(temp_sequence)

            if not sequences:
                print("No long sequences found.")
                return

            sequences_df = df.loc[sequences].copy()

            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
                sequences_df.to_excel(writer, sheet_name="Sequences", index=False)

            print(f"Sequences extracted to a new sheet in {excel_file}")

            # Calculate percentage of sequences in total mismatched rows
            sequence_percentage = (len(sequences_df) / min_rows) * 100 if min_rows > 0 else 0.0
            print(f"\nPercentage of frames in Sequences sheet relative to total: {sequence_percentage:.2f}%\nand the similarity rows is {(100 - sequence_percentage):.2f}%")

        except Exception as e:
            raise ValueError(f"Error extracting sequences: {e}")

    def runner(self):
        self.compare_filtered_rows()


if __name__ == "__main__":
    csv_comparing = CSVComparator()
    csv_comparing.runner()
