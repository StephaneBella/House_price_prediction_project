import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocessing
from src.data_loader import load_config, load_data

def main():
    config = load_config()
    data_path = config['data']['raw_path']
    df = load_data(data_path)
    
    x_train, x_test, y_train, y_test = preprocessing(df)
    print("Prétraitement terminé.")
if __name__ == '__main__':
    main()
