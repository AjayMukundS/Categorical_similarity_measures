from distutils.core import setup
setup(
  name = 'Categorical_similarity_measures',
  packages = ['Categorical_similarity_measures'],
  version = '0.4',
  license='MIT',
  description = 'Similarity Measures Utility Package',
  long_description="Determining similarity or distance between two objects is a key step for several data mining and knowledge discovery tasks. For quantitative data, Minkowski distance plays a major role in finding the distance between two entities. The prevalently known and used similarity measures are Manhattan distance which is the Minkowski distance of order 1 and the Euclidean distance which is the Minkowski distance of order 2. But, in the case of categorical data, we know that there does not exist an innate order and that makes it problematic to find the distance between two categorical points. This is a utility package for finding similarity measures such as Eskin, IOF, OF, Overlap (Simple Matching), Goodall1, Goodall2, Goodall3, Goodall4, Lin, Lin1, Morlini_Zani (S2), Variable Entropy and Variable Mutability. These similarity measures help in finding the distance between two or more objects or entities containing categorical data.",
  long_description_content_type="text/markdown",
  author = 'Ajay Mukund S',
  author_email = 'ajaymukund1998@gmail.com',
  url = 'https://github.com/AjayMukundS/Categorical_similarity_measures',
  download_url = 'https://github.com/AjayMukundS/Categorical_similarity_measures/archive/v_04.tar.gz',
  keywords = ['Similarity', 'Distance', 'Categorical data'],
  install_requires=[
          'pandas',
          'numpy',
          'category_encoders',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
