import sys
import time
import random
import numpy as np
import os

def write_to_file(filename, message):
    output_path = os.path.join('/output', filename)
    with open(output_path, 'a') as f:
        f.write(message + '\n')

def simulate_training(learning_rate, output_file):
    write_to_file(output_file, f"Starting training with learning rate: {learning_rate}")
    epochs = 10
    for epoch in range(1, epochs + 1):
        time.sleep(0.5)  # Simulate some processing time
        loss = random.uniform(0, 1) / (epoch + learning_rate)
        accuracy = random.uniform(0.5, 1) - (0.1 / learning_rate)
        write_to_file(output_file, f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    final_accuracy = random.uniform(0.8, 0.99)
    write_to_file(output_file, f"Training completed. Final accuracy: {final_accuracy:.4f}")
    return final_accuracy

def main():
    if len(sys.argv) != 2:
        print("Usage: python hyperparameter_tuning.py <learning_rate>")
        sys.exit(1)

    try:
        learning_rate = float(sys.argv[1])
    except ValueError:
        print("Error: Learning rate must be a number")
        sys.exit(1)

    output_file = f"output_lr_{learning_rate}.txt"
    write_to_file(output_file, f"Hyperparameter tuning script started with learning rate: {learning_rate}")

    accuracy = simulate_training(learning_rate, output_file)

    write_to_file(output_file, f"Hyperparameter tuning completed for learning rate {learning_rate}")
    write_to_file(output_file, f"Best accuracy achieved: {accuracy:.4f}")
    write_to_file(output_file, f"testing numpy: {np.array([1,2,3])}")

if __name__ == "__main__":
    main()