# Project Structure

```
├── 📁 .github/                    # CI/CD workflows and automation
├── 📁 .venv/                      # Python virtual environment
├── 📁 docker/                     # Docker configurations
├── 📁 k8s/                        # Kubernetes deployment manifests
├── 📁 src/                        # Main source code
├── 📁 tests/                      # Test suites
├── 📄 .dockerignore              # Docker ignore patterns
├── 📄 .gitignore                 # Git ignore patterns
├── 📄 bankruptcy_prediction.ipynb # Original research notebook
├── 📄 bankruptcy_prediction_optimized.ipynb # Optimized notebook
├── 📄 build-and-push.ps1         # Docker build script
├── 📄 CompanyBankruptcyData.csv   # Dataset
├── 📄 CompanyBankruptcyData Documentation.txt # Data documentation
├── 📄 company_bankruptcy_prediction_report.pdf # Original report
├── 📄 config.json                # Configuration file
├── 📄 DEPLOYMENT.md              # Deployment guide
├── 📄 enterprise_report_final.md # Enterprise documentation
├── 📄 PDF_Conversion_Instructions.md # PDF generation guide
├── 📄 README.md                  # Main project documentation
├── 📄 report-style.css           # CSS for professional reports
├── 📄 requirements.txt           # Python dependencies
├── 📄 run_pipeline.py            # Main pipeline script
└── 📄 run_simple.py              # Simple execution script
```

## Clean Project Environment ✅

### Removed Files:
- ❌ `__pycache__/` directories (Python cache)
- ❌ `outputs/` directory (temporary artifacts)
- ❌ `PDF_Generation_Guide.md` (duplicate documentation)
- ❌ `README_OPTIMIZED.md` (duplicate README)
- ❌ `DOCKER_GUIDE.md` (merged into DEPLOYMENT.md)
- ❌ `report.md` (old report version)
- ❌ `generate_pdf_report.py` (complex PDF generator)
- ❌ `simple_pdf_generator.py` (failed PDF generator)
- ❌ `simple_api.py` (experimental API script)
- ❌ `test_api.py` (basic test script)
- ❌ All `.pyc` files (Python compiled files)

### Organized Structure:
- ✅ **Source Code**: All in `src/` directory
- ✅ **Tests**: Comprehensive tests in `tests/` directory
- ✅ **Documentation**: Clear, non-duplicate documentation
- ✅ **Infrastructure**: Docker and Kubernetes configs organized
- ✅ **CI/CD**: GitHub Actions workflows in `.github/`
- ✅ **Configuration**: Proper `.gitignore` and configuration files

### Key Files Retained:
- 📊 **Data**: `CompanyBankruptcyData.csv` and documentation
- 📓 **Notebooks**: Original and optimized versions
- 📋 **Reports**: Original PDF and enterprise markdown version
- ⚙️ **Scripts**: Essential execution and pipeline scripts
- 🏗️ **Infrastructure**: Docker and Kubernetes configurations
- 📚 **Documentation**: Clean, organized documentation

The project environment is now clean, organized, and ready for professional development and deployment.