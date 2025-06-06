from __future__ import unicode_literals, print_function, division
import torch
import jieba
import re
from io import open
import os
import torch.nn as nn
import torch.nn.functional as F
import random

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


# 语言类
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
    # 动态构建路径，基于脚本目录
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本目录: C:\Users\17374\PycharmProjects\pytorch
    data_path = os.path.join(base_dir, '..', 'data', 'PyTorch-11', 'eng-cmn', f'{lang1}-{lang2}.txt')
    data_path = os.path.normpath(data_path)  # 规范化路径
    print(f"尝试读取文件: {data_path}")
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件 '{data_path}' 未找到。")
        lines = open(data_path, encoding='utf-8').read().strip().split('\n')
        print(f"成功读取 {len(lines)} 行")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        raise
    pairs = [[normalizeString_en(s) if i == 0 else normalizeString_cn(s) for i, s in enumerate(l.split('\t'))] for l in
             lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 过滤句子对
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
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        seq_len = encoder_outputs.size(0)
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
        words = list(jieba.cut(sentence))
        print(f"分词结果: {words}")
        return [lang.word2index[word] for word in words if word in lang.word2index]
    words = sentence.split()
    print(f"分词结果: {words}")
    return [lang.word2index[word] for word in words if word in lang.word2index]


def tensorFromSentence(lang, sentence, is_chinese=False):
    indexes = indexesFromSentence(lang, sentence, is_chinese)
    print(f"词汇索引: {indexes}")
    if len(indexes) > MAX_LENGTH - 1:
        indexes = indexes[:MAX_LENGTH - 1]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# 评估函数
def evaluate(encoder, decoder, sentence, input_lang, output_lang, is_chinese=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, is_chinese=is_chinese)
        input_length = input_tensor.size()[0]
        print(f"输入张量形状: {input_tensor.shape}")
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            print(f"解码器输出索引: {topi.item()}")
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            elif topi.item() not in output_lang.index2word:
                decoded_words.append('<UNK>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words


# 训练函数
def train(encoder, decoder, pairs, input_lang, output_lang, n_iters=1000, learning_rate=0.01):
    encoder.train()
    decoder.train()
    optimizer_e = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        pair = random.choice(pairs)  # 随机选择句子对
        input_tensor = tensorFromSentence(input_lang, pair[0], is_chinese=reverse)
        target_tensor = tensorFromSentence(output_lang, pair[1], is_chinese=not reverse)
        loss = 0
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        for di in range(target_tensor.size(0)):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_e.step()
        optimizer_d.step()
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {loss.item() / target_tensor.size(0)}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(encoder.state_dict(), os.path.join(base_dir, 'encoder1.pth'))
    torch.save(decoder.state_dict(), os.path.join(base_dir, 'decoder1.pth'))
    print("模型已保存！")


# 命令行翻译应用
def translation_app(encoder, decoder, input_lang, output_lang, reverse):
    print("欢迎使用中英文翻译应用！")
    print("输入 '退出' 或 'quit' 退出程序。")
    print(f"当前翻译方向: {'中文 → 英文' if reverse else '英文 → 中文'}")
    print("请输入句子：")
    while True:
        try:
            sentence = input("> ").strip()
            if sentence.lower() in ['退出', 'quit']:
                print("退出翻译应用。")
                break
            if not sentence:
                print("请输入有效句子！")
                continue
            output_words = evaluate(encoder, decoder, sentence, input_lang, output_lang, is_chinese=reverse)
            output_sentence = ' '.join([w for w in output_words if w != '<EOS>']) if not reverse else ''.join(
                [w for w in output_words if w != '<EOS>'])
            print("翻译结果:", output_sentence if output_sentence else "<无有效翻译>")
        except KeyError as e:
            print(f"错误：输入包含未知词汇 '{e}'，请尝试其他句子或扩充数据集。")
        except Exception as e:
            print(f"错误：{e}，请检查输入或联系开发者。")


# 主执行
if __name__ == '__main__':
    # 打印调试信息
    print("当前工作目录:", os.getcwd())
    print("脚本目录:", os.path.dirname(os.path.abspath(__file__)))

    # 设置翻译方向
    reverse = True
    try:
        input_lang, output_lang, pairs = prepareData('eng', 'cmn', reverse)
    except FileNotFoundError as e:
        print(f"无法加载数据集: {e}")
        print(
            "请确保文件 'C:\\Users\\17374\\PycharmProjects\\data\\PyTorch-11\\eng-cmn\\eng-cmn.txt' 存在并包含正确格式的句子对。")
        exit(1)

    # 初始化模型
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 加载或训练模型
    base_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.join(base_dir, 'encoder1.pth')
    decoder_path = os.path.join(base_dir, 'decoder1.pth')
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print("加载已保存的模型...")
        encoder1.load_state_dict(torch.load(encoder_path, map_location=device))
        attn_decoder1.load_state_dict(torch.load(decoder_path, map_location=device))
        encoder1.eval()
        attn_decoder1.eval()
    else:
        print("未找到模型，开始训练...")
        train(encoder1, attn_decoder1, pairs, input_lang, output_lang, n_iters=1000)
        print("训练完成，加载模型...")
        encoder1.load_state_dict(torch.load(encoder_path, map_location=device))
        attn_decoder1.load_state_dict(torch.load(decoder_path, map_location=device))
        encoder1.eval()
        attn_decoder1.eval()

    # 调试词汇表
    print(f"输入语言词汇量: {input_lang.n_words}")
    print(f"输出语言词汇量: {output_lang.n_words}")
    print(f"样本输入词汇: {list(input_lang.word2index.keys())[:5]}")
    print(f"样本输出词汇: {list(output_lang.index2word.values())[:5]}")

    # 启动翻译应用
    translation_app(encoder1, attn_decoder1, input_lang, output_lang, reverse)