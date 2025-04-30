import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import json


class Plotting:
    def __init__(self):
        
    
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")

        with open(config_path) as f:
            config = json.load(f)
            self.config = config["plotting"]

        self.valid_path = os.path.join(os.getcwd(), "output_files", self.config["valid_path"])
        self.non_valid_path = os.path.join(os.getcwd(), "output_files", self.config["non_valid_path"])
        self.data_3D_file = os.path.join(os.getcwd(), self.config["data_3D_file"])
        self.file = self.config["file"]
        self.trial_num = self.config["trial_num"]
        self.plotting_type = self.config["plotting_type"]


    def interpolate_trials(self):
        """
        Groups data by trial, finds the group with the maximum frames,
        and applies linear interpolation to match the max size.
        Ensures first and last frames remain in place, but interpolates everything else.
        """

        print("Reading from path:", self.data_3D_file)
        df = pd.read_csv(self.data_3D_file, skiprows=3)
        df = df.iloc[:, 0:4]
        df.iloc[:, 1:4] = df.iloc[:, 1:4].astype(float)
        for i in range(1, 4):
            df.iloc[:, i] = df.iloc[:, i].interpolate(method='linear', limit_direction='both') \
                                        .fillna(method='bfill') \
                                        .fillna(method='ffill')
            

        if df.iloc[:, 1:4].isna().any().any():
            print("Warning: NaNs still present in 3D data after interpolation.")
        else:
            print("3D data interpolated successfully.")
    

        if self.file.lower() == 'valid':
            file_1 = pd.read_csv(self.valid_path)

        elif self.file.lower() == 'non_valid':
            file_1 = pd.read_csv(self.non_valid_path)

            file_1 = file_1[file_1.iloc[:, 3] == 1].copy()
            

        if 'trial' not in file_1.columns:
            raise ValueError("The dataset must contain a 'trial' column.")


        grouped = file_1.groupby('trial')
        max_trial_size = max(grouped.size())  
        print(f"Max Trial Size: {max_trial_size}")

        interpolated_data = [] 

        df_indexed = df.set_index(df.columns[0])
        trial_3D_groups = {}

        for trial, group in grouped:
            frame_numbers = group.iloc[:, 0].tolist()
            trial_3D_data = df_indexed.loc[frame_numbers].reset_index()
            trial_3D_groups[trial] = trial_3D_data


            original_x = list(trial_3D_data.iloc[:, 1].values)
            original_y = list(trial_3D_data.iloc[:, 2].values)
            original_z = list(trial_3D_data.iloc[:, 3].values)

            original_size = len(original_x)

            num_missing = max_trial_size - original_size

            if num_missing > 0:
                
                first_x, last_x = original_x[0], original_x[-1]
                first_y, last_y = original_y[0], original_y[-1]
                first_z, last_z = original_z[0], original_z[-1]

                # remove first and last rows
                inner_x = original_x[1:-1]
                inner_y = original_y[1:-1]
                inner_z = original_z[1:-1]

                
                step_size = (len(inner_x) + 1) / (num_missing + 1)
                insert_positions = [int(i * step_size) for i in range(1, num_missing + 1)]
                

                for idx in reversed(insert_positions):
                    inner_x.insert(idx, np.nan)
                    inner_y.insert(idx, np.nan)
                    inner_z.insert(idx, np.nan)

                new_x = [first_x] + inner_x + [last_x]
                new_y = [first_y] + inner_y + [last_y]
                new_z = [first_z] + inner_z + [last_z]


            else:
                new_x = original_x
                new_y = original_y
                new_z = original_z


            new_x = pd.Series(new_x).interpolate(method='linear', limit_direction='both') \
                        .fillna(method='bfill') \
                        .fillna(method='ffill') \
                        .tolist()

            new_y = pd.Series(new_y).interpolate(method='linear', limit_direction='both') \
                    .fillna(method='bfill') \
                    .fillna(method='ffill') \
                    .tolist()

            new_z = pd.Series(new_z).interpolate(method='linear', limit_direction='both') \
                    .fillna(method='bfill') \
                    .fillna(method='ffill') \
                    .tolist()
            
            for x, y, z in zip(new_x, new_y, new_z):
                interpolated_data.append({'trial': trial, 'X': x, 'Y': y, 'Z': z})


        return pd.DataFrame(interpolated_data)

    
    def plot_trials(self, df):
        trials = df['trial'].unique()
        x_col, y_col, z_col = 'X', 'Y', 'Z'

        if type(self.trial_num) != str:
            if self.trial_num not in trials:
                print(f"Trial {self.trial_num} not found in the dataset.")
                return
            trials = [self.trial_num]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d') if self.plotting_type == '3D' else plt.gca()

        all_x = []
        all_y = []
        all_z = []

        first_label = True

        for trial in trials:
            trial_data = df[df['trial'] == trial]
            x_vals = trial_data[x_col].values
            y_vals = trial_data[y_col].values
            z_vals = trial_data[z_col].values

            # if np.isnan(x_vals).any() or np.isnan(y_vals).any() or np.isnan(z_vals).any():
            #     print(f"Warning: NaN values found in trial {trial}. Skipping this trial.")
            #     continue

            window_length = 25
            x_vals = savgol_filter(x_vals, window_length=window_length, polyorder=2)
            y_vals = savgol_filter(y_vals, window_length=window_length, polyorder=2)
            z_vals = savgol_filter(z_vals, window_length=window_length, polyorder=2)

            all_x.append(x_vals)
            all_y.append(y_vals)
            all_z.append(z_vals)

            if self.plotting_type == '2D':
                ax.plot(x_vals, y_vals, color='black', alpha=0.5)
            else:
                ax.plot(x_vals, y_vals, z_vals, color='black', alpha=0.5)
                
            start_label = "Start" if first_label else ""
            end_label = "End" if first_label else ""


            first_label = False

            if self.plotting_type == '2D':
                ax.scatter(x_vals[0], y_vals[0], color='cyan', s=50, label=start_label)
                ax.scatter(x_vals[-1], y_vals[-1], color='red', s=50, label=end_label)
            else:
                ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='cyan', s=50, label=start_label)
                ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='red', s=50, label=end_label)

        
        if type(self.trial_num) == str:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            all_z = np.array(all_z)
            avg_x = np.mean(all_x, axis=0)
            avg_y = np.mean(all_y, axis=0)
            avg_z = np.mean(all_z, axis=0)

            if self.plotting_type == '2D':
                ax.plot(avg_x, avg_y, color='purple', linewidth=2, label="Average Trajectory")
            else:
                ax.plot(avg_x, avg_y, avg_z, color='purple', linewidth=2, label="Average Trajectory")

        ax.set_title("Trajectory of Trials" if type(self.trial_num) == str else f"Trajectory of Trial {self.trial_num}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if self.plotting_type == '3D':
            ax.set_zlabel("Z Coordinate")

        ax.legend()
        ax.grid(True)


        # Only invert Y axis for 2D plots
        if self.plotting_type == '2D':
            ax.invert_yaxis()

        plt.show()

    
    def runner(self):
        df_interpolated = self.interpolate_trials()
        self.plot_trials(df_interpolated)

        if type(self.trial_num) == str:
            for trial in df_interpolated['trial'].unique():
                self.trial_num = trial
                self.plot_trials(df_interpolated)

        return df_interpolated
            

if __name__ == "__main__":
    plot = Plotting()
    interpolated_data = plot.runner()
    