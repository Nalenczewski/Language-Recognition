![](./README_files/char.jpg)

# Chinese Pinyin Mapping Function
_______________
## Contents
1. **Introduction**
2. **My Approach**
3. **Findings**
4. **Ideas for Further Research**
5. **Recommendations**

## 1. Introduction
____
The Chinese language is written using symbols called characters to represent meaning. Unlike words in a phonetic alphabet, there is no indication of how to pronounce each character. Therefore, to help people learn Chinese, a phonetic alphabet based on the latin alphabet was invented called pinyin. In different contexts, the same character can have different pronunciations, with some characters having as many as seven different pronunciations.

In textbooks, when a new character is presented, the pinyin is usually also given so that the reader learns how to pronounce the character. Chinese texts in the wild typically only contain characters. But for learners of the language, it is sometimes helpful to give the pronunciation of each character in a text. It is for this reason that I created a function that maps each character to it’s correct pronunciation given the context. 

## 2. My Approach
____
The first step in the process of building the pinyin mapping function was to generate a list of all the Chinese characters with all of their pronunciations and their frequencies. This required some parsing and pruning of a list of each character with its given pronunciation found at the [Unihan Database](https://www.unicode.org/charts/unihan.html). Unihan are Unicode for Chinese characters.

Some characters were missing frequency information so for these characters, I gave them a frequency of 1. Most of the characters that had multiple pronunciations had frequencies for them.

The second step was to build a mapping function that mapped single characters to their pronunciation based on the most frequent pronunciation of a given character. This approach would lead to correct mappings most of the time, but there are still some cases where the mapping would be wrong based on context. For this reason a more sophisticated mapping function was needed.

The final step in building a function that outputs the correct pinyin of each character was to use a phrase dictionary which contained the correct pronunciation of 96,809 words and phrases. I should note here that most Chinese words are made up of two to three characters. The majority of words and phrases in this list were two-letter words.

For the second function, I first looked at each set of two consecutive characters in the text and compared them to all of the words in the phrase dictionary to see if there was a match (ie. if these two characters were a word with a set pronunciation). If there was a match, the function output that pronunciation, if there was no match, then the pronunciation from the pinyin frequency dictionary was used. Then the function moves on to the next character in the text to repeat the process.

### Sample input and output
___

![](./README_files/Input.png)

![](./README_files/Output.png)

## 3. Findings
___

In order to evaluate how my pinyin mapping function worked, I passed it three texts and checked them to see if there were any mispronunciations. Below are some of the pronunciations that were incorrect.

**Article 1: New York Times Chinese Edition - [Ukraine](https://cn.nytimes.com/world/20220226/ukraine-russia-war-kyiv/)**

Errors:

![](./README_files/SS1.png)
Should be fourth tone: wèi

![](./README_files/SS2.png)
Should be second tone: wéi

Total errors: 2

**Article 2: Wikipedia - [Lenovo](https://zh.wikipedia.org/wiki/%E8%81%94%E6%83%B3%E9%9B%86%E5%9B%A2)**

Errors:

![](./README_files/SS3.png)
Should be second tone: wéi

![](./README_files/SS4.png)
Should be second tone: lúo

![](./README_files/SS5.png)
Should be second tone: wéi

Total errors: 3

**Article 3: Baidu - [Chinese History](https://baijiahao.baidu.com/s?id=1718023695121784419&wfr=spider&for=pc)**

Errors:

![](./README_files/SS6.png)
Should be third tone: chǔ

![](./README_files/SS7.png)
Should be second tone: wéi

Total errors: 2

Besides these few pronunciations, all of the other pronunciations were correct.

## 4. Ideas for Further Research
___

In order to build a more precise pinyin mapping function, the function could be made to check three-letter and four-letter words and phrases first using the phrase dictionary, and then proceeding to check for two-letter words and then single-characters words like in the function I created.

One could also present the pinyin and character of the output in a more attractive way. [This project](https://github.com/jeffreyxuan/toneoz-font-pinyin-kai) on Github presents some options for doing this.

My function only deals with simplified Chinese characters, which are used primarily in mainland China. But other places like Taiwan and Hong Kong use traditional Chinese characters, so a function could be built that would take these other forms of characters into account. Not all characters have a simplified version, by some estimates, 30% of the 3,500 most common characters in use have been simplified.

I could find a way to eliminate the incorrect pronunciations I found, especially wéi/wèi, which was responsible for over half of the errors I found.


## 5. Recommendations

The pinyin creator can be used to give the correct pronunciations to any simplified Chinese text passed to it. Three ways I recommend using this tool are: 
* It can be used to study the language by giving readers the pronunciations of unknown characters.
* It can be integrated into an app or dictionary.
* It can be used to cross check other pinyin mapping texts, apps, and dictionaries.

