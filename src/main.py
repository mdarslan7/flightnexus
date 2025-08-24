from src.train_model import train_model
from src.critical_flights import find_critical_flights

def main():

    print("--- Starting Backend Pipeline ---")
    print("\nStep 1: Training delay prediction model...")
    train_model()
    print("\nStep 2: Finding critical flights...")
    find_critical_flights()
    print("\n--- Backend Pipeline Complete ---")

if __name__ == '__main__':
    main()