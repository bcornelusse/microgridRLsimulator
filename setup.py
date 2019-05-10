from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='microgridRLsimulator',
      version='',
      description='',
      url='',
      author='',
      author_email='',
      license='',
      packages=packages,
      install_requires=[
        'python-dateutil', 'docopt==0.6.2', 'matplotlib==3.0.2', 'numpy==1.15.4',
        'pandas==0.23.4', 'scipy==1.1.0', 'tensorflow==1.12.0', 'tflearn==0.3.2', 'sphinx', 'gym'
      ],
      zip_safe=False)
