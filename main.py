import subprocess
import sys

def main():
    print("🎨 GAN Assistant")
    print("1. Train GAN on CIFAR-10")
    print("2. Generate image from trained model")
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        print("\n🚀 Starting GAN training...")
        subprocess.run([sys.executable, "train.py"])
    elif choice == "2":
        print("\n🎨 Generating image from trained generator...")
        subprocess.run([sys.executable, "inference.py"])
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()2