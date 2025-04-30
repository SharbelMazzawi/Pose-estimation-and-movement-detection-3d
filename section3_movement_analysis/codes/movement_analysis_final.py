import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
from pathlib import Path
import time
import json



class MovementAnalyzer:
    def __init__(self):
                
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")

        with open(config_path) as f:
            config = json.load(f)
            self.config = config['movement_analysis']

        self.column_groups = self.config["column_groups"] if self.config["column_groups"] else {}
        self.thresholds = self.config["thresholds"] 
        self.min_length = self.config["min_length"]
        self.padd_trial = self.config["padd_trial"]
        self.image_path = self.config["image_path"]
        self.ref_idx = self.config["reference_idx"]
        
        self.save_dir = os.path.join(os.getcwd(), 'output_files')
        self.file_path = os.path.join(os.getcwd(), "output_files", self.config["file_path"])
        self.video_path = os.path.join(os.getcwd(), self.config["video_path"]) 
        self.data_start_row = self.config["data_start_row"]
        self.frames_folder = os.path.join(os.getcwd(), self.config["frames_folder"])
        
        
        self.data = None
        self.ref_points = {}
        self.movement_flags = {}
        self.likelihood_sums = {}
        self.groups = [g for g in self.column_groups if g.lower() != 'food'] if self.column_groups else []
        
        
        self._load_data()
        self._get_reference_points()
    

    def _load_data(self):
        """Load and prepare data from CSV."""
        self.data = pd.read_csv(self.file_path, header=None, low_memory=False).apply(pd.to_numeric, errors='coerce')
        self.data = self.data.iloc[self.data_start_row:].reset_index(drop=True)


    def _select_ref_point_on_img(self):
        """Let user select point on image."""
        point = None
        finished = False
        
        def on_click(event, x, y, flags, img):
            if event == cv2.EVENT_LBUTTONDOWN:
                nonlocal point
                point = (x, y)
                img_copy = img.copy()
                cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Image', img_copy)

        img = cv2.imread(self.image_path)
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', on_click, img)
        print("Click to select point. Press 'f' to finish.")

        while not finished:
            if (cv2.waitKey(1) & 0xFF) == ord('f'):
                finished = True
        
        cv2.destroyAllWindows()
        return point


    def _get_reference_points(self):
        """Get reference points either from image or data."""
        if self.image_path:
            for group in self.column_groups:
                print(f"Select reference for {group}")
                self.ref_points[group] = np.array(self._select_ref_point_on_img())
        else:
            print(self.column_groups.keys())
            self.ref_points = {
                g: self.data.iloc[self.ref_idx, idxs[:-1]].values.astype(float) 
                for g, idxs in self.column_groups.items()}
            

    def _find_movements(self):
        """Calculate movement flags for all groups."""
        
        for group, indices in self.column_groups.items():
            group_data = self.data.iloc[:, indices[:-1]].astype(float)
            likelihood = (self.data.iloc[:, indices[-1]].astype(float)  #the indices[-1] is the liklehood column.
                        if len(indices) > 2 else np.ones(len(self.data))) # this because we dont need the food liklehood so we did not give 3 columns for food as input.
            
            flag = ((group_data > (self.ref_points[group] + self.thresholds[group])) | 
                    (group_data < (self.ref_points[group] - self.thresholds[group]))).any(axis=1).astype(int)

            
            if group.lower() == 'food':
                self.movement_flags[group] = 1 - flag
            else:
                self.movement_flags[group] = flag


            # gets the liklehood of the frames that stand the first condition (the threshold)
            self.likelihood_sums[group] = likelihood * flag


    # this function takes the groups of ones (continuous groups) we got from the threshold_distance_condition
    def _get_runs(self, arr):
        """Get indices of continuous 1s."""
        diffs = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return [list(range(s, e)) for s, e in zip(starts, ends)]
    

    # function that handles the cases where the model predict movements of hand for both hands (in the same frames).
    # i did handle this by the liklehood, the hand that has more liklehood wins.
    def _resolve_overlaps(self, g1, g2, g1_runs, g2_runs):
        """Handle overlaps between movement hand groups."""

        prev_g1 = []
        prev_g2 = []

        for g1_run in g1_runs[:]:
            for g2_run in g2_runs[:]:
                overlap = set(g1_run) & set(g2_run)
                if overlap:
                    overlap = list(overlap)
                    if self.likelihood_sums[g1][overlap].sum() >= self.likelihood_sums[g2][overlap].sum():
                        self.movement_flags[g2][g2_run] = 0
                        if g2_run != prev_g2:
                            g2_runs.remove(g2_run)
                            prev_g2 = g2_run 
                    else:
                        self.movement_flags[g1][g1_run] = 0
                        if g1_run != prev_g1:
                            g1_runs.remove(g1_run)
                            prev_g1 = g1_run   



    # fills small gaps between founded trails 
    def _fill_small_gaps(self, group, runs):
        """Fill gaps of less than 5 zeros between sequences of ones in movement flags."""

        for i in range(len(runs)-1):  # Iterate through runs
            gap_start = runs[i][-1] #+ 1  # Start of the zero gap
            gap_end = runs[i + 1][0] #- 1  # End of the zero gap
            
            if ((gap_end - gap_start + 1) < 4) and (len(runs[i])>5 and len(runs[i+1])>5):
                # If gap length is less than 5, change zeros to one
                self.movement_flags[group][gap_start:gap_end + 1] = 1



    # applying more side conditions to filter movements groups we got from the _find_movements
    def _filter_movements(self, group, runs):
        """Filter movements based on length."""

        for run in runs:
            if len(run) < self.min_length:
                self.movement_flags[group][run] = 0
                continue
            
    

    def _adding_padd(self, group, runs, padd):
        """
        Add padding to the movement runs and update movement flags accordingly.
        """
        for i in range(len(runs)):
            start = runs[i][0] - padd
            end = runs[i][-1] + padd
            
            # Ensure indices are within bounds
            start = max(0, start)
            end = min(len(self.movement_flags[group]) - 1, end)
            
            # Flatten the list of indices and use .iloc to assign the values
            # padded_indices = list(range(start, end + 1))
            padded_indices = [idx for idx in range(start, end + 1) if idx < len(self.movement_flags[group])]

            # Update the movement flags
            self.movement_flags[group].iloc[padded_indices] = 1
            
            
                        

    def _assign_trials(self, g1, g2):
        """Assign trial numbers to movements."""
        trials = np.zeros(len(self.data))
        trial_num = 1
        in_trial = False

        for i in range(len(trials)):
            if self.movement_flags[g1][i] == 1 or self.movement_flags[g2][i] == 1:
                if not in_trial:
                    in_trial = True
                trials[i] = trial_num
            else:
                if in_trial:
                    trial_num += 1
                in_trial = False

        return trials
    


    def _update_validity(self, movement_groups):
        """Ensure all frames in a movement sequence take the validity of the first frame."""
        
        valid = valid = np.zeros(len(self.data))
        # Identify movement groups (continuous nonzero trials)
        for group_of_1s in movement_groups:
            first_frame = group_of_1s[0]  # First frame in the sequence
            
            # Check if the first frame meets the original validity condition
            food = next((key for key in self.movement_flags.keys() if key.lower() == 'food'), None)
            right_hand = next((key for key in self.movement_flags.keys() if key.lower() == 'right_hand'), None)
            is_valid = (self.movement_flags[food][first_frame] == 1 & self.movement_flags[right_hand][first_frame] == 1)
            

            valid[group_of_1s] = int(is_valid)

        return valid



    def analyze(self):
        """Perform complete movement analysis and save results."""
        # Calculate flags and process movements
        self._find_movements() 

        g1, g2 = self.groups

        g1_runs = self._get_runs(self.movement_flags[g1])
        g2_runs = self._get_runs(self.movement_flags[g2])

        self._resolve_overlaps(g1, g2, g1_runs, g2_runs)

        self._fill_small_gaps(g1, g1_runs)
        self._fill_small_gaps(g2, g2_runs)

        self._filter_movements(g1, g1_runs)
        self._filter_movements(g2, g2_runs)

        self._adding_padd(g1, g1_runs, self.padd_trial)
        self._adding_padd(g2, g2_runs, self.padd_trial)
        
        
        trials = self._assign_trials(g1, g2)

        movement_groups = self._get_runs(trials != 0)
        valid = self._update_validity(movement_groups)
        
        # Save results
        results = {
            'frame_number': self.data.index,
            **self.movement_flags,
            'trial': trials,
            'valid': valid
        }


        output_path = os.path.join(self.save_dir, "results.csv")
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f" Analysis complete. Saved to {output_path}")


    def save_valid_invalid_movements(self):
        """
        Create two CSV files:
        1. "valid_movement.csv" - containing valid movements (valid == 1).
        2. "non_valid_movement.csv" - containing trials that exist (trial != 0) but are invalid (valid == 0).
        """
            
        results_path = os.path.join(self.save_dir, "results.csv")
        results_df = pd.read_csv(results_path)
        
        # Separate valid and invalid movements
        valid_df = results_df[results_df["valid"] == 1].copy()
        invalid_df = results_df[(results_df["trial"] != 0) & (results_df["valid"] == 0)].copy()
            
        # get the original data (df_linear_interpolation.csv)
        original_data = pd.read_csv(self.file_path, header=None, low_memory=False)
        original_data = original_data.iloc[self.data_start_row:].reset_index(drop=True)
        
        # Extract relevant indices (excluding likelihood columns)
        relevant_columns = []
        for group, indices in self.column_groups.items():
            if group.lower() != 'food':
                relevant_columns.extend(indices[:-1]) # columns indices of the groups without the likelihood column
        
        ## get original columns names from row 1 (adjust as needed)
        # column_names_row = original_data.iloc[1]
        # column_names = [column_names_row[idx] for idx in relevant_columns]  
        
        self.valid_df = valid_df
        self.invalid_df = invalid_df
        
        # Change the trial number to 1 up to len 
        self.valid_df["trial"] = pd.factorize(self.valid_df["trial"])[0] + 1
        self.invalid_df["trial"] = pd.factorize(self.invalid_df["trial"])[0] + 1
        
        # Function to process movement data and save to CSV
        def process_movement_data(df, output_filename, add_right_movement=False):
            movement_data = []
            for _, row in df.iterrows():
                frame_idx = int(row["frame_number"])  # Get the frame index
                trial_number = row["trial"] # Get the trial number
                food_in_place = row[next((key for key in df.keys() if key.lower() == 'food'), None)] # Get the "food in place" (0 or 1)
                    
                # Extract relevant data from the original data
                selected_data = original_data.iloc[frame_idx, relevant_columns].values
                
                # Check if "right_hand" movement exists in results.csv
                right_movement = 1 if (frame_idx in results_df["frame_number"].values and 
                                    results_df.loc[results_df["frame_number"] == frame_idx, "Right_hand"].values[0] == 1) else 0
                
                # Append row with new ordering
                if add_right_movement:
                    movement_data.append([frame_idx, trial_number, food_in_place, right_movement] + list(selected_data))
                else:
                    movement_data.append([frame_idx, trial_number, food_in_place] + list(selected_data))
            
            # Convert to DataFrame and save as CSV
            if add_right_movement:
                columns = ["frame_number", "trial", "food_in_place", "right_movement", 'right_center_x', 'right_center_y', 'left_center_x', 'left_center_y']
            else:
                columns = ["frame_number", "trial", "food_in_place", 'right_center_x', 'right_center_y', 'left_center_x', 'left_center_y']
                
            movement_df = pd.DataFrame(movement_data, columns=columns)
            movement_path = os.path.join(self.save_dir, output_filename)
            movement_df.to_csv(movement_path, index=False)
            print(f" {output_filename} saved to {movement_path}")
            
        # Save valid movements
        process_movement_data(valid_df, "valid_movement.csv")
        
        # Save invalid trials with the additional "right_movement" column
        process_movement_data(invalid_df, "non_valid_movement.csv", add_right_movement=True)




    def extract_frames(self):
        """
        Extract all frames from a video using its original FPS and save them to a folder.
        If the folder already exists and contains the correct number of frames, the function exits.

        Returns:
            None
        """
        video_path = os.path.join(os.getcwd(), self.video_path)
        save_frames_folder = os.path.join(os.getcwd(), 'extracted_frames')

        # Open the video file to count total frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count from video

        # Check if the folder exists and has the correct number of frames
        if os.path.exists(save_frames_folder):
            existing_frames = len([f for f in os.listdir(save_frames_folder) if f.endswith('.jpg')])

            if existing_frames == total_frames:
                print(f"Frames already extracted ({existing_frames}/{total_frames}). Exiting function.")
                cap.release()
                return  # Exit function early if the frames are already saved

        # Create folder if it doesn't exist
        if not os.path.exists(save_frames_folder):
            print(f"Creating frames folder: {save_frames_folder}")
            os.makedirs(save_frames_folder, exist_ok=True)

        # Extract frames and save them
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  

            # Save the frame to the output folder
            frame_filename = os.path.join(save_frames_folder, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1

        cap.release()
        print(f"Frames extracted and saved to {save_frames_folder}")


    
    
    def generate_movement_video(self, output_video="output_files", frame_rate=50):
        """
        Generate a video from movement frames and save everything inside the output_files folder.

        Parameters:
        - output_video (str, optional): Folder where the video and movement frames should be saved.
                                        Defaults to "output_files".
        - frame_rate (int): Frame rate of the output video.
        """
        # check of folder exist 
        if not os.path.exists(self.frames_folder):
            print(f"Frames folder not found: {self.frames_folder}")
            self.extract_frames()
        

        self.save_dir = os.path.join(os.getcwd(), output_video)

        # Ensure the output directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Generate the default video filename inside self.save_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.save_dir, f"movement_video_{timestamp}.mp4")

        
        movement_frames = self.valid_df["frame_number"].values
    
        frame_list_path = os.path.join(self.save_dir, "movement_frames.txt")
        with open(frame_list_path, "w") as f:
            f.write("\n".join(map(str, movement_frames)))

        print(f" Saved movement frame indices to {frame_list_path}")

        # Ensure the frames_folder exists
        if not self.frames_folder:
            print(" No frames folder specified. Cannot generate video.")
            return
        
        # Load a sample frame to determine video size
        sample_frame_path = Path(self.frames_folder) / f"frame_{movement_frames[0]}.jpg"
        sample_frame = cv2.imread(str(sample_frame_path))
        if sample_frame is None:
            print(" Could not load sample frame. Check frame file paths.")
            return

        height, width, _ = sample_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        print(" Creating movement video...")
        idx = 0
        for frame_idx in tqdm(movement_frames, desc="Processing frames"):
            frame_path = Path(self.frames_folder) / f"frame_{frame_idx}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                # add text to the frame
                cv2.putText(frame, f"Frame: {frame_idx}, Trial: {self.valid_df['trial'].iloc[idx]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                if frame is not None:
                    out.write(frame)
                    idx += 1
            else:
                print(f" Frame {frame_idx} not found!")

        out.release()
        print(f" Movement video saved to {video_path}")

    def generate_full_video(self, output_video="output_files", frame_rate=180):
        """
        Generate a video from all frames in the folder while displaying:
        - Frame number
        - Trial number (updates when a valid movement starts)
        
        Parameters:
        - output_video (str, optional): Folder where the video should be saved.
        - frame_rate (int): Frame rate of the output video.
        """
        # Ensure the frames folder exists
        if not os.path.exists(self.frames_folder):
            print(f"Frames folder not found: {self.frames_folder}")
            self.extract_frames()

        # Set output folder
        self.save_dir = os.path.join(os.getcwd(), output_video)
        os.makedirs(self.save_dir, exist_ok=True)

        # Generate unique video filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.save_dir, f"full_video_{timestamp}.mp4")

        # Get all frame filenames in the folder
        frame_files = sorted([f for f in os.listdir(self.frames_folder) if f.endswith(".jpg")],
                            key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Load a sample frame to determine video size
        sample_frame = cv2.imread(os.path.join(self.frames_folder, frame_files[0]))
        if sample_frame is None:
            print("Could not load sample frame. Check frame file paths.")
            return

        height, width, _ = sample_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        # Get valid frame data
        valid_frames = dict(zip(self.valid_df["frame_number"], self.valid_df["trial"]))

        print("Creating full video with trial tracking...")

        trial_counter = 0  # Initialize trial counter
        for frame_file in tqdm(frame_files, desc="Processing frames"):
            frame_idx = int(frame_file.split("_")[1].split(".")[0])
            frame_path = os.path.join(self.frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            # Determine trial number: Keep the previous trial unless we hit a valid frame
            if frame_idx in valid_frames:
                trial_counter = valid_frames[frame_idx]

            # Overlay text
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Trial: {trial_counter}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Write frame to video
            out.write(frame)

        out.release()
        print(f"Full video saved to {video_path}")




    def runner(self):
        self.analyze()
        self.save_valid_invalid_movements()
        self.extract_frames()
        self.generate_movement_video()
        self.generate_full_video()
        
            
            

if __name__ == "__main__":
    movement_detection = MovementAnalyzer()
    movement_detection.runner()

