
## App Details
>This App showcases a RNN based neural network to automatically generates novel but synthetic TV script fragments based on [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) sit-com. The model has been trained on a small subset of the Seinfeld TV sit-com's script from 9 seasons.

The App is composed of two tabs, "ABOUT" and "SCRIPT WRITER".

### ABOUT
This page, detailing the app usage.

### SCRIPT WRITER
Accepts a text prompt/seed (prime-word) and the length (number of words) of the script to be generated. corpus (TV/film script, book, etc.). "Generate" button press will kick-off the scrip generation process to generate a text snippet that'd look like if this generated text came from the original script/book. 

<img src="https://github.com/sssingh/tv-script-generation-rnn/blob/master/assets/writer-1.png?raw=true"><br><br>
<img src="https://github.com/sssingh/tv-script-generation-rnn/blob/master/assets/writer-2.png?raw=true">

Note that the prime-word must be a word thats present in vocabulary of training text 
corpus. For reference the vocabulary can be downloaded directly from the writer tab.

Note that even though Seinfeld script is used to train and generate the text, any other text corpus (tv/film script, books, text corpus) can be used to re-train the model and generate the relevant text. There could be multiple use cases where the same text generation technique can be used.


## Project Source
👉 [Visit GitHub Repo](https://github.com/sssingh/tv-script-generation-rnn)

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

# Contact Me
[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sunil@sunilssingh.me)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/@thesssingh)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sssingh/)
[![website](https://img.shields.io/badge/web_site-8B5BE8?style=for-the-badge&logo=ko-fi&logoColor=white)](https://sunilssingh.me)

## Credits
- Dataset used in this project is provided by [Udacity](https://www.udacity.com/)
- Dataset is a subset taken from Kaggle, [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
- GRU, RNN diagrams courtesy [Daniel V. Godoy](https://github.com/dvgodoy)

