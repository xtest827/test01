from __future__ import unicode_literals, print_function, division
import torch
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import jieba
import random
import re
from io import open
from matplotlib import font_manager
import matplotlib.ticker as ticker
import os
from torch.amp import GradScaler, autocast

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10  # 最大序列长度

# 语言类，用于构建词汇表
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addSentence_cn(self, sentence):
        for word in jieba.cut(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# 规范化函数
def normalizeString_en(s):
    s = s.lower().strip()
    s = re.sub(r'([.!?])', r'\1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

def normalizeString_cn(s):
    s = s.strip()
    return s

# 读取语言对
def readLangs(lang1, lang2, reverse=False):
    print('Reading lines')
    try:
        lines = open(f'data/PyTorch-11/eng-cmn/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file 'data/PyTorch-11/eng-cmn/{lang1}-{lang2}.txt' not found.")
    pairs = [[normalizeString_en(s) if i == 0 else normalizeString_cn(s) for i, s in enumerate(l.split('\t'))] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

# 过滤句子对，确保长度不超过 MAX_LENGTH
def filterPairs(pairs, reverse):
    def sentence_length(s, is_chinese=False):
        if is_chinese:
            return len(list(jieba.cut(s)))
        return len(s.split())
    return [p for p in pairs if sentence_length(p[0], is_chinese=reverse) <= MAX_LENGTH and
            sentence_length(p[1], is_chinese=not reverse) <= MAX_LENGTH]

# 准备数据
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')
    pairs = filterPairs(pairs, reverse)
    print(f'Trimmed to {len(pairs)} sentence pairs')
    for pair in pairs:
        if reverse:
            input_lang.addSentence_cn(pair[0])
            output_lang.addSentence(pair[1])
        else:
            input_lang.addSentence(pair[0])
            output_lang.addSentence_cn(pair[1])
    print('Counted words:')
    print(f'{input_lang.name}: {input_lang.n_words}')
    print(f'{output_lang.name}: {output_lang.n_words}')
    return input_lang, output_lang, pairs

# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 注意力解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, 1)  # 输出单个数值，动态计算权重
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        seq_len = encoder_outputs.size(0)
        # 动态计算注意力权重
        attn_weights = torch.zeros(seq_len, device=device)
        for i in range(seq_len):
            attn_weights[i] = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(attn_weights, dim=0).view(1, 1, -1)
        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights.squeeze(0)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 句子转张量
def indexesFromSentence(lang, sentence, is_chinese=False):
    if is_chinese:
        return [lang.word2index[word] for word in jieba.cut(sentence) if word in lang.word2index]
    return [lang.word2index[word] for word in sentence.split() if word in lang.word2index]

def tensorFromSentence(lang, sentence, is_chinese=False):
    indexes = indexesFromSentence(lang, sentence, is_chinese)
    if len(indexes) > MAX_LENGTH - 1:  # 留空间给 EOS
        indexes = indexes[:MAX_LENGTH - 1]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(input_lang, output_lang, pair, reverse):
    input_tensor = tensorFromSentence(input_lang, pair[0], is_chinese=reverse)
    target_tensor = tensorFromSentence(output_lang, pair[1], is_chinese=not reverse)
    return input_tensor, target_tensor

# 训练步骤，使用混合精度
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    scaler = GradScaler('cuda')
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 动态分配 encoder_outputs 大小
    encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)
    loss = 0

    with autocast('cuda'):
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = random.random() < 0.5

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

    scaler.scale(loss).backward()
    scaler.step(encoder_optimizer)
    scaler.step(decoder_optimizer)
    scaler.update()
    return loss.item() / target_length

# 时间格式化
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%dm %ds (- %dm %ds)" % (s // 60, s % 60, rs // 60, rs % 60)

# 训练循环，带早停机制
def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromPair(input_lang, output_lang, random.choice(pairs), reverse)
                     for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{timeSince(start, iter / n_iters)} ({iter} {iter / n_iters * 100:.1f}%) {print_loss_avg:.4f}')
            if print_loss_avg < 0.01:  # 放宽早停条件
                print(f"Early stopping at iteration {iter}, loss {print_loss_avg:.4f}")
                break

# 评估函数
def evaluate(encoder, decoder, sentence, input_lang, output_lang, is_chinese=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, is_chinese=is_chinese)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, input_length)

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di, :decoder_attention.size(-1)] = decoder_attention
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

# 随机评估
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang, is_chinese=reverse)
        output_sentence = ' '.join(output_words[:-1]) if not reverse else ''.join(output_words[:-1])
        print('<', output_sentence)
        print('')

# 注意力可视化
def showAttention(input_sentence, output_words, attentions):
    try:
        myfont = font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc')
    except:
        print("Warning: Chinese font not found, using default font.")
        myfont = font_manager.FontProperties()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    input_words = list(jieba.cut(input_sentence)) if reverse else input_sentence.split()
    ax.set_xticklabels([''] + input_words + ['<EOS>'], fontproperties=myfont, rotation=90)
    ax.set_yticklabels([''] + output_words, fontproperties=myfont)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence,
                                       input_lang, output_lang, is_chinese=reverse)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words[:-1]) if not reverse else ''.join(output_words[:-1]))
    showAttention(input_sentence, output_words, attentions)

# 主执行
if __name__ == '__main__':
    # 设置 reverse=True 表示中文 -> 英文；reverse=False 表示英文 -> 中文
    reverse = True
    input_lang, output_lang, pairs = prepareData('eng', 'cmn', reverse)
    print(random.choice(pairs))

    # 初始化模型
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 检查保存的模型
    base_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.join(base_dir, 'encoder1.pth')
    decoder_path = os.path.join(base_dir, 'decoder1.pth')
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print("Loading saved models...")
        encoder1.load_state_dict(torch.load(encoder_path, map_location=device))
        attn_decoder1.load_state_dict(torch.load(decoder_path, map_location=device))
        encoder1.eval()
        attn_decoder1.eval()
    else:
        print("No saved models found, training new models...")
        trainIters(encoder1, attn_decoder1, n_iters=10000, print_every=1000)
        torch.save(encoder1.state_dict(), encoder_path)
        torch.save(attn_decoder1.state_dict(), decoder_path)
        print(f"Models saved to {encoder_path} and {decoder_path}")

    # 评估
    evaluateRandomly(encoder1, attn_decoder1)
    # 对于 reverse=True（中文 -> 英文），使用中文输入
    evaluateAndShowAttention("我们在讨论你的未来。")
    # 对于 reverse=False（英文 -> 中文），取消下面ение