import os
import pandas as pd
import numpy as np
import cv2
import json

class DataProcessor:
    def __init__(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")

        with open(config_path) as f:
            config = json.load(f)
            self.config = config["data_process"]

        print('The location for saving files and reading files is: ', os.getcwd())  
        self.df_dlc_oneCam = pd.read_csv(os.path.join(os.getcwd(),  self.config["file_name"]))
        
        self.save_dir = os.path.join(os.getcwd(), 'output_files')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)


    def process_columns_with_pcutoff(self, target_string ='likelihood', 
                                    ):
        """
        Processes columns in a CSV file by identifying those containing a specific string value,
        starting from the row after the one that contains the target string, converting all cells
        to numeric, and replacing numeric values below a given cutoff with NaN, as well as the 
        specified number of preceding columns in the same row.
        
        If there are fewer columns than the specified 'coordinates' value, an error is raised.

        Args:
            input_path (str): Path to the input CSV file.
            target_string (str): The string value to search for in columns.
            pcutoff (float): The cutoff value. Numeric values below this will result in the 
                            preceding columns in the same row being replaced with NaN.
            coordinates (int): Number of preceding columns to set to NaN if the condition is met. 2 in the 2D case
            output_path (str, optional): Path to save the modified DataFrame. If None and save=True, creates a default file.
            save (bool, optional): Whether to save the modified DataFrame. Default is False.
        
        Returns:
            pd.DataFrame: The modified DataFrame.
        """
        
        df = self.df_dlc_oneCam
        pcutoff = self.config['pcutoff']
        self.coordinates = self.config['coordinates']
        
        # Check if the DataFrame has enough columns
        if self.coordinates >= len(df.columns):
            raise ValueError(f"The DataFrame does not have enough columns for 'coordinates={self.coordinates}'. "
                            f"Only {len(df.columns)} columns are available.")

        # Iterate over all columns
        for column in df.columns:
            # Check if the column contains the target string
            rows_with_target = df[column].apply(lambda x: isinstance(x, str) and x == target_string)
            if rows_with_target.any():
                print(f"Column '{column}' contains the target string '{target_string}'. Processing...")

                # Get the first row index where the target string is found
                target_row_index = rows_with_target.idxmax()

                # Convert rows after the target string to numeric
                df.loc[target_row_index + 1:, column] = pd.to_numeric(df.loc[target_row_index + 1:, column], errors='coerce')

                # Apply the pcutoff condition to numeric values and replace preceding cells
                for row_idx in range(target_row_index + 1, len(df)):
                    if pd.notna(df.at[row_idx, column]) and df.at[row_idx, column] < pcutoff:
                        # Ensure there are enough preceding columns
                        column_idx = df.columns.get_loc(column)
                        if column_idx < self.coordinates:
                            raise ValueError(
                                f"Cannot apply 'coordinates={self.coordinates}' because column '{column}' "
                                f"does not have enough preceding columns."
                            )

                        # Replace the preceding columns (up to the specified number) in the same row with NaN
                        for offset in range(1, self.coordinates + 1):
                            df.iat[row_idx, column_idx - offset] = np.nan


        df.to_csv(os.path.join(self.save_dir, 'df_pcut.csv'), index=False)
        print(f"Modified DataFrame saved as df_pcut")
        self.df_pcut = df

        return df

    def linear_interpolation(self):
        """
        Applies linear interpolation to numeric NaN values for all columns while preserving
        the first three rows if they contain non-numeric values.
        
        The function detects missing (NaN) values and applies interpolation only where needed.
        """
        
        df = self.df_pcut
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True) # for the upload of the csv... old issue 
        
        first_three_rows = df.iloc[:3]
        numeric_part = df.iloc[3:].apply(pd.to_numeric, errors='coerce')

        numeric_part.interpolate(method='linear', limit_direction='both', inplace=True)

        # Reconstruct the full DataFrame
        df = pd.concat([first_three_rows, numeric_part]).reset_index(drop=True)

        output_path = os.path.join(self.save_dir, 'df_linear_interpolation.csv')
        df.to_csv(output_path, index=False)
        
        print(f" Updated DataFrame with interpolated values saved as df_linear_interpolation.csv")

        return df


    def dataCleaner(self):
        df = self.df_dlc_oneCam
        col_names = df.iloc[0] + "_" + df.iloc[1]
        data = df[3:].applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        self.df_dlc_oneCam_cleaned = pd.DataFrame(data.values, columns=col_names)   
            
    def runner(self):
        self.dataCleaner()
        self.process_columns_with_pcutoff()
        self.linear_interpolation()
            

if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.runner()
    data_processor.extract_frames()
