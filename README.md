<div align="center">

# Jefferson Conza

**Mathematics Student & Machine Learning Engineer**  
*Bridging Mathematics and AI.*

[Website & Essays](https://jeffersonconza.github.io) • [Projects Portfolio](https://jeffersonconza.github.io/projects) • [Interactive Math & AI Lab](https://jeffersonconza.github.io/matematica-ia) • [LinkedIn](https://www.linkedin.com/in/jefferson-conza-46779b59/) • [Email](mailto:jeffconza3@gmail.com)

</div>

---

### 🔬 Profile & Background

- 📐 **Mathematics** senior student at [Universidad Yachay Tech](https://yachaytech.edu.ec/) (Ecuador).
- 🤖 **Machine Learning Engineer** trained via [Anyone AI](https://www.anyoneai.com/) & [Udacity](https://www.udacity.com/).
- ☁️ **Cloud Application Developer** candidate at [AWS Cloud Institute](https://aws.amazon.com/).
- 🎯 **Research & Engineering Focus**: Scientific Machine Learning (PINNs), Computer Vision for Bio-Imaging, Transformers & NLP, and Applied Statistics.

---

### 🚀 Selected Projects

- **[SentimentScope: Transformer NLP Architecture](https://github.com/JeffersonConza/sentimentscope-transformer-nlp)**  
  Modular 4-layer Transformer encoder built from scratch in PyTorch (multi-head self-attention, positional encoding, custom embeddings). Trained with AdamW on GPU with 100% test pass rate CI/CD.  
  `PyTorch` `Hugging Face` `Pytest` `CI/CD` • [Project Dossier](https://jeffersonconza.github.io/projects/sentimentscope-transformer-nlp/)

- **[CheX-DS: Pneumonia Detection with Transformers](https://github.com/JeffersonConza/pneumonia-detection-cnn-xray)**  
  Hybrid architecture (DenseNet-121 + Swin Transformer) for pediatric chest X-ray screening. Achieved **99.0% recall** with Weighted Asymmetric Loss and Grad-CAM interpretability.  
  `PyTorch` `Swin Transformer` `Streamlit` `Docker` • [Project Dossier](https://jeffersonconza.github.io/projects/deteccion-neumonia-chex-ds/)

- **[PINNs & SciML Laboratory](https://jeffersonconza.github.io/projects/laboratorio-pinns-aws/)**  
  Continuous operator learning framework solving 1D Burgers, Heat Diffusion, and Navier-Stokes equations with physics conservation loss, automatic differentiation, and AWS S3/EC2 deployment.  
  `PyTorch` `DeepXDE` `FastAPI` `Docker` `AWS` • [Project Dossier](https://jeffersonconza.github.io/projects/laboratorio-pinns-aws/)

- **[3D Isotropic Bio-Image Reconstruction](https://jeffersonconza.github.io/projects/reconstruccion-isotropica-bioimagen/)**  
  Unpaired image-to-image translation with 3D CycleGAN in PyTorch to resolve axial point-spread-function (PSF) distortion in confocal fluorescence microscopy.  
  `PyTorch` `CycleGAN` `Bio-Imaging` `LaTeX` • [Project Dossier](https://jeffersonconza.github.io/projects/reconstruccion-isotropica-bioimagen/)

- **[Energy Forecast Hub](https://github.com/JeffersonConza/energy-forecast-hub)**  
  Production-grade time-series forecasting microservice with XGBoost and Random Forest, orchestrated via FastAPI and interactive Streamlit UI.  
  `Python` `FastAPI` `Streamlit` `Docker` `XGBoost` • [Project Dossier](https://jeffersonconza.github.io/projects/centro-pronostico-energia/)

---

### 🧪 Interactive Mathematics & AI Lab

A computational laboratory dedicated to interactive numerical simulations and theoretical foundations on [jeffersonconza.github.io/matematica-ia](https://jeffersonconza.github.io/matematica-ia):

- 🎯 **Continuous Dynamic Programming**: Value function derivation and policy convergence in Bellman optimality.
- 🔠 **QKV Multi-Head Attention**: Vectorized scaled dot-product attention simulator in latent projection spaces.
- 📐 **17 Stewart Equations & Millennium Problems**: Rigorous mathematical breakdowns of fundamental laws and open conjectures.

---

### 💻 Physics-Informed Neural Network (PINN) Operator

```python
import torch

def physics_loss(u_net, x, t, nu=0.01 / torch.pi):
    """
    1D Viscous Burgers' equation residual:
    f := ∂u/∂t + u (∂u/∂x) - ν (∂²u/∂x²) = 0
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = u_net(torch.cat([x, t], dim=1))
    
    # Exact differential operators computed via PyTorch automatic differentiation
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual ** 2)
```

---

### 🛠️ Technical Stack

- **Languages**: Python, R, Julia, SQL, C++, Bash, LaTeX
- **Deep Learning & SciML**: PyTorch, DeepXDE, Hugging Face, Torchvision, TensorFlow, scikit-learn
- **Scientific Computing**: NumPy, SciPy, pandas, SymPy, Matplotlib, Manim
- **Cloud & Deployment**: Docker, AWS (EC2, S3, SageMaker), FastAPI, Streamlit, Git, Linux

---

### ✍️ Selected Writing

- [Cosmos: Alexander von Humboldt y la Red de la Vida](https://jeffersonconza.github.io/blog/alexander-von-humboldt-cosmos-red-de-la-vida/)
- [Telescopio Espacial Nancy Grace Roman & Energía Oscura](https://jeffersonconza.github.io/blog/telescopio-espacial-nancy-grace-roman-energia-oscura/)
- [PINNs: Redes Neuronales Informadas por la Física](https://jeffersonconza.github.io/blog/pinns-redes-neuronales-informadas-por-la-fisica/)
- [Puntos de Lagrange: Mecánica Celeste y Equilibrio Orbital](https://jeffersonconza.github.io/blog/puntos-de-lagrange-mecanica-celeste-equilibrio-orbital/)

*Explore all bilingual essays and notes at [jeffersonconza.github.io/blog](https://jeffersonconza.github.io/blog).*

---

<div align="center">
<sub>Designed with minimalism and mathematical clarity • © Jefferson Conza</sub>
</div>
