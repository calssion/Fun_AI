# QA systems(问答系统)
### tutorial(指导)：
[https://rajpurkar.github.io/mlx/qa-and-squad/](https://rajpurkar.github.io/mlx/qa-and-squad/)  

[https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507](https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507)  

[https://github.com/aswalin/SQuAD](https://github.com/aswalin/SQuAD)  

Question answering is an important NLP task and longstanding milestone for artificial intelligence systems. QA systems allow a user to ask a question in natural language, and receive the answer to their question quickly and succinctly.(问答是一项重要的NLP任务，是人工智能的长久基石。问答系统让用户用自然语言提问，然后能快速有效地给予回复)  

The ability to read a piece of text and then answer questions about it is called reading comprehension. Reading comprehension is challenging for machines, requiring both understanding of natural language and knowledge about the world.(阅读部分文本然后回答的能力称为阅读理解。阅读理解对于机器而言是一大挑战，不仅需要对自然语言的理解，还需要来自于世界的知识)  

![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fuocphh38uj20fx0ay758.jpg)  

![image](https://wx4.sinaimg.cn/mw1024/8311c72dly1fuocsyods5j20mc0femxy.jpg)   

### SQuAD Dataset(数据集)  
Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage. With 100,000+ question-answer pairs on 500+ articles, SQuAD is significantly larger than previous reading comprehension datasets.(斯坦福问答数据集是一个新的阅读理解数据集，由群体工作者在一组维基百科文章中提出的问题，其中每个问题的答案是一段文本或跨度，从相应的阅读段落。100000 +问答对500 +文章，斯坦福问答数据集明显大于以前的阅读理解数据集。)   

### Problem(问题)  
For each observation in the training set, we have a context, question, and text.(对于训练集的每一次观察，我们有内容、问题和文本)  
![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuoczspv1nj20wb0dcabq.jpg)   

The goal is to find the text for any new question and context provided. This is a closed dataset meaning that the answer to a question is always a part of the context and also a continuous span of context. I have broken this problem into two parts for now(目标是在任意新问题和提供的内容中找到对应文本。这是一个相关的数据集，回复往往是内容的一部分和一段连续的跨度。我将这个问题分为两个部分) -   
1、Getting the sentence having the right answer (highlighted yellow)(获取含有正确答案的内容(图中黄色标注))   
2、Once the sentence is finalized, getting the correct answer from the sentence (highlighted green)(一旦内容准备好了，获取其正确答案(图中绿色标注))   

## Introducing Infersent, Facebook Sentence Embedding(介绍句子嵌入)   
These days we have all types of embeddings word2vec, doc2vec, food2vec, node2vec, so why not sentence2vec. The basic idea behind all these embeddings is to use vectors of various dimensions to represent entities numerically, which makes it easier for computers to understand them for various downstream tasks. (现如今我们有各类嵌入方式，所以也能有sentence2vec句子嵌入。这些嵌入背后的基本思想是使用不同维度的向量去表示实体数据，这使得计算机在各种下游任务中更容易去理解它们。)    

Traditionally, we used to average the vectors of all the words in a sentence called the bag of words approach. Each sentence is tokenized to words, vectors for these words can be found using glove embeddings and then take the average of all these vectors. This technique has performed decently, but this is not a very accurate approach as it does not take care of the order of words. (传统方式，我们通常把一个句子中所有单词的向量平均称为词袋法。每个句子被标记到单词，这些单词的向量可以使用嵌入来找到，然后取所有这些向量的平均值。这项技术已经执行得很好，但这不是一个非常精确的方法，因为它不关心单词的顺序。)    

### sentence embeddings (句子嵌入(内容嵌入))  
Create a vocabulary from the training data and use this vocabulary to train infersent model. Once the model is trained, provide sentence as input to the encoder function which will return a 4096-dimensional vector irrespective of the number of words in the sentence. (从训练数据中创建词汇，并使用该词汇训Infersent模型。一旦模型被训练好了，提供句子作为输入到编码器函数，它将返回一个4096维向量，而不管句子中的单词数量。)    

tried using sentence embedding to solve the first part of the problem described in the previous section (尝试使用句子嵌入来解决我们上面提到的第一个问题)-   

Break the paragraph/context into multiple sentences. The two packages that I know for processing text data are - Spacy & Textblob. I have used package TextBlob for the same. It does intelligent splitting, unlike spacy’s sentence detection which can give you random sentences on the basis of period. (拆分段落/内容成众多句子。我所知的两个库可以做到的是Spacy & Textblob。 我试过使用TextBlob达到相同效果，不同于在初期给定任意句子的宽大句子检测)   

![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fuodtjvi66j20n00do3zx.jpg)    

![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fuoduezw72j20mi08jjrw.jpg)   

Get the vector representation of each sentence and question using Infersent model(用Infersent模型获取每个句子和问题的向量)   

Create features like distance, based on cosine similarity and Euclidean distance for each sentence-question pair(对于每一对句子和问题，创建基于余弦相似度和欧式距离的特征)   

## Models(模型)
I have further solved the problem using two major methods (我使用了两种主要的方法来解决这个问题)-   

Unsupervised Learning where I am not using the target variable. Here, I am returning the sentence form the paragraph which has the minimum distance from the given question(无监督学习，我不使用标签。在这里，我返回那些与问题具有最小距离的段落的句子)   

Supervised Learning - Creation of training set has been very tricky for this part, the reason being the fact that there is no fixed number of sentences in each part and answer can range from one word to multiple words. The only paper I could find that has implemented logistic regression is by the Stanford team who has launched this competition & dataset. They have used multinomial logistic regression explained in this paper and have created 180 million features (sentence detection accuracy for this model was 79%), but it is not clear how they have defined the target variable. (有监督学习-训练集的创建对于这部分很微妙，原因是这里没有对于每一部分句子的合适的数字，而且答案可能横跨多个单词。我能找到的唯一的论文是斯坦福团队对于竞赛和数据集使用了逻辑回归。在论文中解释道，他们用了多变量逻辑回归，创建了1.8亿个特征(模型的句子检测准确率是79%)，但他们是如何定义标签的并不清晰。)    

### Unsupervised Learning Model(无监督学习模型)
Here, I first tried using euclidean distance to detect the sentence having minimum distance from the question. The accuracy of this model came around 45%. Then, I switched to cosine similarity and the accuracy improved from 45% to 63%. This makes sense because euclidean distance does not care for alignment or angle between the vectors whereas cosine takes care of that. Direction is important in case of vectorial representations.(这里，我首先尝试了用欧式距离检测与问题具有最小距离的句子。模型准确率达到大约45%。然后，我转成使用余弦相似度，模型准确率提高到63%。这是必然的，因为欧式距离并不关心向量间的排列或角度，但余弦相似度关心。方向对于向量化表示是挺重要的。)    

But this method does not leverage the rich data with target labels that we are provided with. However, considering the simple nature of the solution, this is still giving a good result without any training. I think the credit for the decent performance goes to Facebook sentence embedding.(但这个方法没有平衡我们提供的具有标签的大量数据。然而，考虑到解决的简单，这对于没有大量训练的情况下，仍不失为一种好方法。我认为句子嵌入是对好的表现的保证。)   

### Supervised Learning Models(有监督模型)
Here, I have transformed the target variable form text to the sentence index having that text. For the sake of simplicity, I have restricted my paragraph length to 10 sentences (around 98% of the paragraphs have 10 or fewer sentences). Hence, I have 10 labels to predict in this problem.(这里，我把标签从文本转成含有该文本的句子索引。为了简单实现，我限制了段落长度最多为10个句子。这样在这个问题上，我有10个标签来预测了。)    

For each sentence, I have built one feature based on cosine distance. If a paragraph has less number of sentences, then I am replacing it’s feature value with 1 (maximum possible cosine distance) to make total 10 sentences for uniformity. It will be easier to explain this process with an example.(对于每一个句子，我建立了一个基于余弦相似度的特征。如果一个段落的句子很少，我会用值1来替换它的特征来使得全部10个句子均匀(1为最大的余弦距离)。它让表现一个例子的进度更加容易)   

### Example(例子)
- **Question**：“To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?”   
- **Context**：“Architecturally, the school has a Catholic character. Atop the Main Building\’s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend “Venite Ad Me Omnes”. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. **It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.** At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.”   
- **Text**：“Saint Bernadette Soubirous”   

In this case, the target variable will become 5, because that’s the index of the bolded sentence. We will have 10 features each corresponding to one sentence in the paragraph. The missing values for column_cos_7, column_cos_8, and column_cos_9 are filled with 1 because these sentences do not exists in the paragraph(在这个案例中，标签值为5。我们有10个特征值对应于段落里的每一个句子。缺失值column_cos_7, column_cos_8 和  column_cos_9将会被1填充，因为这些句子不存在于这个段落里。)    

## Dependency Parsing(依存句法分析)   
Another feature that I have used for this problem is the “Dependency Parse Tree”. This is marginally improving the accuracy of the model by 5%. Here, I have used Spacy tree parsing as it is has a rich API for navigating through the tree.(另一个我用的新特征是依存句法分析树。这能提升模型准确度5%。这里，我使用Spacy的树分析，因为它对于树导向有丰富的API。)    

![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fuoli239hnj20m807wwen.jpg)    
![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fuoli22z6tj20om0e6aad.jpg)    

Relations among the words are illustrated above the sentence with directed, labeled arcs from heads to dependents. We call this a Typed Dependency structure because the labels are drawn from a fixed inventory of grammatical relations. It also includes a root node that explicitly marks the root of the tree, the head of the entire structure. (字词之间的关系在句子中用指向的、标记的弧从头到从属关系来说明。我们称这是一种类型化的依赖结构，因为标签是从固定的语法关系清单中提取出来的。它还包括根节点，该根节点显式地标记树的根、整个结构的头部。)    

The idea is to match the root of the question which is “appear” in this case to all the roots/sub-roots of the sentence. Since there are multiple verbs in a sentence, we can get multiple roots. If the root of the question is contained in the roots of the sentence, then there are higher chances that the question is answered by that sentence. Considering that in mind, I have created one feature for each sentence whose value is either 1 or 0. Here, 1 represents that the root of question is contained in the sentence roots and 0 otherwise. (思路是匹配问题的根和出现在句子中的所有根与子根。因为一个句子中含有大量的单词，所以一个句子有大量的根。如果问题的根包含在句子的根中，那极有可能问题的答案在该句子中。考虑到这点，我为每一个句子创建了一个特征，值为1或0。这里，1代表问题的根包含在句子中，否则为0。)    

we have 20 features in total combining cosine distance and root match for 10 sentences in a paragraph. The target variable ranges from 0 to 9. (我们有20个特征结合段落中10个句子的余弦相似度和根匹配。标签值为从0到9。)   
![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fupg651ekrj20e90cwgm3.jpg)    
![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fupg651xkfj20fa0d8wez.jpg)    

