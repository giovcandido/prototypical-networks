from setuptools import setup, find_packages

# Load long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Configure setup
setup(
    name = 'protonets',
    version = '1.0.0',
    author = 'Giovani Candido',
    author_email = 'giovcandido@outlook.com',
    license = 'GNU General Public License v3.0',
    description = (
        'Vanilla and Prototypical Networks with Random Weights for '
        'image classification on Omniglot and mini-ImageNet.'
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/giovcandido/prototypical-networks-project',
    packages = ['protonets']
)
