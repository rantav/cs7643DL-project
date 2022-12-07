import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/output/style_transfered/results.csv')

df = df[['style_weight', 'style_accuracy', 'content_accuracy']]
df.index = df['style_weight']
df = df.drop(columns=['style_weight'])
df.plot(style='o-', logx=True)
plt.suptitle('Style Transfered images Content and Style Accuracy')
plt.title('with Gram Matrix classifier')
plt.show()