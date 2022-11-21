
import paddle
from paddlenlp.transformers import BertModel

class bertLinear(paddle.nn.Layer):
    def __init__(self,config):
        super(bertLinear, self).__init__()
        self.config = config
        self.model_name = "bertLinear"
        self.bert = BertModel.from_pretrained(config.bert_name)
        self.dropout = paddle.nn.Dropout(0.5)
        self.linear = paddle.nn.Linear(in_features=self.config.bert_hid_size,out_features=self.config.label_num)
    def forward(self, bert_inputs):
        attention_mask = bert_inputs.not_equal(paddle.to_tensor(0,dtype="int64"))
        attention_mask = attention_mask.astype("int64")
        sequence_output , cls_output = self.bert(bert_inputs, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        outputs = self.linear(sequence_output)
        return outputs

class bertLstm(paddle.nn.Layer):
    def __init__(self,config):
        super(bertLstm, self).__init__()
        self.config = config
        self.model_name = "bertLstm"
        self.bert = BertModel.from_pretrained(config.bert_name)
        self.lstm = paddle.nn.LSTM(input_size=config.bert_hid_size
                                   ,hidden_size=config.bert_hid_size
                                   )
        self.linear = paddle.nn.Linear(in_features=config.bert_hid_size
                                       ,out_features=config.label_num
                                       )
    def forward(self, bert_inputs):
        attention_mask = bert_inputs.not_equal(paddle.to_tensor(0, dtype="int64"))
        attention_mask = attention_mask.astype("int64")
        sequence_output, cls_output = self.bert(bert_inputs,attention_mask=attention_mask)
        outputs = self.lstm(sequence_output)
        outputs = self.linear(outputs)
        return outputs

class bertCNN(paddle.nn.Layer):
    def __init__(self,config):
        super(bertCNN, self).__init__()
        self.config = config
        self.model_name = "bertCNN"
        self.bert = BertModel.from_pretrained(self.config.bert_name)
        self.base = paddle.nn.Sequential(
            paddle.nn.Dropout(0.1)
            ,paddle.nn.Conv2D(in_channels=self.config.bert_hid_size
                              ,out_channels=self.config.conv_hid_size
                              ,kernel_size=1
                              )
            ,paddle.nn.GELU()
        )
        self.conv2d = paddle.nn.Conv2D(in_channels=self.config.bert_hid_size
                         , out_channels=self.config.conv_hid_size
                         , kernel_size=1
                         )
        self.conv = paddle.nn.LayerList([
                paddle.nn.Conv2D(in_channels=self.config.conv_hid_size
                                 ,out_channels=self.config.conv_hid_size
                                 ,dilation=d
                                 ,groups=self.config.conv_hid_size
                                 ,kernel_size=3
                                 ,padding=d
                                 )  for d in self.config.dilation])
        self.linear = paddle.nn.Linear(in_features=self.config.conv_hid_size * len(config.dilation),
                                      out_features=self.config.label_num)
        self.dropout = paddle.nn.Dropout(0.5)

    def forward(self, bert_inputs):
        attention_mask = bert_inputs.not_equal(paddle.to_tensor(0, dtype="int64"))
        attention_mask = attention_mask.astype("int64")
        sequence_output, cls_output = self.bert(bert_inputs, attention_mask=attention_mask)
        sequence_output = paddle.transpose(paddle.unsqueeze(sequence_output,1),[0,3,1,2])
        sequence_output = self.base(sequence_output)
        # sequence_output = self.conv2d(sequence_output)
        conv_outputs = list()
        for conv in self.conv:
            conv_output = conv(sequence_output)
            conv_output = paddle.nn.functional.gelu(conv_output)
            conv_outputs.append(conv_output)
        conv_output = paddle.concat(conv_outputs,axis=1)
        conv_output = paddle.transpose(conv_output,[0,2,3,1])
        conv_output = paddle.squeeze(conv_output,1)
        outputs = self.linear(conv_output)
        return outputs
