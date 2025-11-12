# GitHub Codespaces Usage Guide for SNNtrainer3D

This guide will help you get started with SNNtrainer3D using GitHub Codespaces.

## What is GitHub Codespaces?

GitHub Codespaces provides a complete, configurable dev environment in the cloud. It allows you to run SNNtrainer3D directly in your browser without any local installation.

## Getting Started

### Step 1: Launch Codespace

1. **Option A - Quick Launch:**
   - Click the "Open in GitHub Codespaces" badge on the main README
   
2. **Option B - From Repository:**
   - Navigate to the [SNNtrainer3D repository](https://github.com/jurjsorinliviu/SNNtrainer3D)
   - Click the green **"Code"** button
   - Select the **"Codespaces"** tab
   - Click **"Create codespace on main"**

### Step 2: Wait for Environment Setup

The first time you launch, Codespaces will:
- Build a Docker container with Python 3.12
- Install all dependencies from [`requirements.txt`](requirements.txt:1)
- Create necessary directories
- Configure the development environment

**Expected wait time:**
- First launch: 2-3 minutes
- Subsequent launches: 10-30 seconds (cached)

### Step 3: Start the Application

Once the environment is ready:

1. **Open the terminal** (it should open automatically, or press `` Ctrl+` `` / `` Cmd+` ``)
2. **Run the application:**
   ```bash
   python app.py
   ```
3. **Access the web interface:**
   - VSCode will automatically detect the Flask server on port 5000
   - A notification will appear: "Your application running on port 5000 is available"
   - Click **"Open in Browser"** or **"Open in Preview"**

Alternatively, go to the **"PORTS"** tab in VSCode and click the globe icon next to port 5000.

## Using SNNtrainer3D in Codespaces

### Workflow

1. **Design Your Network:**
   - Add/remove layers using the interface
   - Configure neuron types (LIF, Lapicque, Synaptic, Alpha, Realistic Lapicque)
   - Set hyperparameters (learning rate, beta, epochs, etc.)

2. **Download Dataset:**
   - Click "Download MNIST Dataset" or select XOR dataset
   - First download may take a moment

3. **Train Your Model:**
   - Click "Train Model"
   - Monitor progress with the progress bar
   - View real-time 3D visualization of your network architecture

4. **Download Results:**
   - After training completes, download the trained weights
   - Files are saved in your Codespace and can be downloaded to your local machine

### Accessing Files

- **View files:** Use the Explorer panel on the left
- **Download files:** Right-click any file → "Download"
- **Upload files:** Drag and drop into the Explorer panel

### Managing Your Codespace

#### Stopping Your Codespace
- Codespaces automatically stop after 30 minutes of inactivity
- Manual stop: Click your Codespace name (bottom-left) → "Stop Current Codespace"

#### Resuming Your Codespace
- Return to the repository and click "Code" → "Codespaces"
- Your previous Codespace will be listed
- Click on it to resume (starts in ~10 seconds)

#### Deleting Your Codespace
If you're done and want to free up resources:
1. Go to [github.com/codespaces](https://github.com/codespaces)
2. Find your SNNtrainer3D codespace
3. Click the three dots → "Delete"

## Troubleshooting

### Port 5000 Not Forwarding
If you don't see the application:
1. Check the **"PORTS"** tab in VSCode
2. Ensure port 5000 is listed
3. Right-click port 5000 → "Port Visibility" → "Public"
4. Click the globe icon to open

### Installation Issues
If dependencies fail to install:
```bash
pip install -r requirements.txt --force-reinstall
```

### Codespace Running Slowly
- Free tier Codespaces have limited resources
- Consider upgrading to 4-core machine type:
  - Click your Codespace name → "Change Machine Type"
  - Select a larger instance

### Dataset Download Fails
If MNIST download fails:
```bash
mkdir -p /tmp/data/mnist
python -c "from Neural_Network import loadMNIST; loadMNIST()"
```

## Tips for Effective Use

### 1. Save Your Work
- Trained models are automatically saved in your Codespace
- Download important results regularly
- Commit changes to your fork if you've modified code

### 2. Resource Management
- Stop Codespace when not in use (free tier: 60 hours/month)
- Monitor usage at [github.com/settings/billing](https://github.com/settings/billing)

### 3. Customization
You can modify the Codespace configuration:
- Edit [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json:1) to change settings
- Edit [`.devcontainer/post-create.sh`](.devcontainer/post-create.sh:1) to add setup steps

### 4. Collaborative Work
- Share your Codespace URL with collaborators (if permissions allow)
- Fork the repository to make your own modifications

## GitHub Codespaces Pricing

### Free Tier
- **Personal accounts:** 120 core-hours/month (60 hours on 2-core)
- **Storage:** 15 GB/month

### Pro/Team Accounts
- **Pro:** 180 core-hours/month
- **Team/Enterprise:** Custom limits

See [GitHub Codespaces pricing](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces) for details.

## Advanced Features

### Using GPU (Paid Feature)
For faster training, you can use GPU-enabled Codespaces:
1. This requires GitHub Team or Enterprise
2. Select a GPU machine type when creating Codespace
3. PyTorch will automatically use CUDA if available

### Persistent Storage
Your Codespace persists:
- Code changes
- Installed packages
- Trained models
- Configuration

Data is kept for 30 days after deletion.

## Getting Help

- **Issues:** Report bugs on [GitHub Issues](https://github.com/jurjsorinliviu/SNNtrainer3D/issues)
- **Documentation:** See main [README.md](README.md:1)
- **Paper:** [SNNtrainer3D Research Paper](https://www.mdpi.com/2076-3417/14/13/5752)
- **Video:** [YouTube Demonstration](https://www.youtube.com/watch?v=UHwPItZTjEs)

## Local Development vs Codespaces

| Feature | Local Development | GitHub Codespaces |
|---------|------------------|-------------------|
| Setup Time | Manual installation | Automated (2-3 min) |
| Platform | OS-dependent | Works anywhere |
| Resources | Your machine | Cloud compute |
| Cost | Free | Free tier available |
| Collaboration | Difficult | Easy sharing |
| Portability | Limited | Browser-based |

## Next Steps

Now that you're familiar with Codespaces:
1. Try training a simple XOR network
2. Experiment with different neuron types
3. Train on MNIST dataset
4. Visualize your trained network architecture
5. Download and analyze the results

Happy training SNNs! 🧠⚡