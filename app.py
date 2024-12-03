from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model and the dataframe for reference data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the form on the homepage
    return render_template('index.html', company_options=df['Company'].unique(),
                           type_options=df['TypeName'].unique(),
                           cpu_options=df['Cpu Brand'].unique(),
                           gpu_options=df['Gpu Brand'].unique(),
                           os_options=df['os'].unique(),
                           resolutions=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
                                        '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                           ram_options=[2, 4, 6, 8, 12, 16, 24, 32, 64],
                           hdd_options=[0, 128, 256, 512, 1024, 2048],
                           ssd_options=[0, 8, 128, 256, 512, 1024])

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    company = request.form['company']
    type = request.form['type']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    ips = 1 if request.form['ips'] == 'Yes' else 0
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    gpu = request.form['gpu']
    os = request.form['os']

    # Compute ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Prepare query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Predict the price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Return the result
    return render_template('index.html',company_options=df['Company'].unique(),
                           type_options=df['TypeName'].unique(),
                           cpu_options=df['Cpu Brand'].unique(),
                           gpu_options=df['Gpu Brand'].unique(),
                           os_options=df['os'].unique(),
                           resolutions=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
                                        '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                           ram_options=[2, 4, 6, 8, 12, 16, 24, 32, 64],
                           hdd_options=[0, 128, 256, 512, 1024, 2048],
                           ssd_options=[0, 8, 128, 256, 512, 1024], prediction_text=f'The predicted price of this configuration is Rs. {predicted_price}')

if __name__ == "__main__":
    app.run(debug=True)
