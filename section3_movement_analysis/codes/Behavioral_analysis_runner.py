from data_processor_Class_final import DataProcessor
from movement_analysis_final import MovementAnalyzer
from comparsion_final import CSVComparator
from plotting_final import Plotting
import os


if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())
    data_processor = DataProcessor()
    data_processor.runner()
    movement_analysis = MovementAnalyzer()  
    movement_analysis.runner()
    csv_comparing = CSVComparator()
    csv_comparing.runner()
    plotting = Plotting()
    plotting.runner()
    