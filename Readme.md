# How to Run the Script

Follow these steps to set up and run the script:

1. **Install a virtual environment:**

   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment:**

   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set the `BRAINTRUST_API_KEY` environment variable:**

   ```sh
   export BRAINTRUST_API_KEY=your_api_key_here
   ```

5. **Run the script:**

   ```sh
   python main.py
   ```

Make sure to replace `your_api_key_here` with your actual Braintrust API key.
