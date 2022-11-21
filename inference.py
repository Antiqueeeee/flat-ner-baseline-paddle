import data_loader
from tqdm import tqdm
import json
import argparse
import models
import os
import paddle
class Predictor(object):
    def __init__(self, model,config):
        self.model = model
        self.config = config
    def load(self, path=None):
        if path:
            self.model.set_state_dict(paddle.load(path))

        else:
            self.model.set_state_dict(paddle.load(
                os.path.join(self.config.save_path, self.config.dataset, self.model.model_name + ".pt")
            ))

    def sequence_tag2tag(self, text, label):
        # 统计序列标注中的实体及其类型
        # 存放实体相关信息，以字典结构保存，其中包括entity、type以及index
        item = dict()
        # 保存当前正在读取的实体，实体结束后会存入item["entity"]中
        _entity = str()
        # ner中存放当前语料包含的所有实体
        ner = list()
        index = list()
        # 遍历序列标注形式的标签，如果当前标签中包含“B-”则表明“上一个实体已经读取完毕，现在开始要开始读取一个新的实体”
        # 如果当前标签中包含“I-”，说明正在读取的实体还未结束，将当前标签所对应的字添加进_entity中，继续遍历
        # 循环结束后，如果item中不为空，说明存在有未保存的实体，将相关实体信息添加到字典中，最后添加到数据集中。
        for i, (t, l) in enumerate(zip(text, label)):
            if "B-" in l:
                if item:
                    item["entity"] = _entity
                    item["index"] = index
                    ner.append(item)
                    _entity = str()
                    item = dict()
                    index = list()
                item["type"] = l.split("-")[1]
                _entity = t
                index.append(i)
            if "I-" in l and item is not None:
                _entity += t
                index.append(i)
        if item:
            item["entity"] = _entity
            item["index"] = index
            ner.append(item)
            _entity = str()
            item = dict()
            index = list()
        return ner
    def predcit(self,data_loader,origin_data):
        result = list()
        batch = 0
        with paddle.no_grad():
            for data_batch in tqdm(data_loader):
                sentence_batch = origin_data[batch : batch + args.batch_size]
                data_batch = [data for data in data_batch]
                bert_inputs, sent_length = data_batch
                outputs = self.model(bert_inputs)
                for sentence, pred_label, bert_input in zip(sentence_batch,outputs,bert_inputs):
                    sentence = sentence["sentence"]
                    _valid_index = bert_input.not_equal(paddle.to_tensor(0, dtype="int64")).astype("int64")
                    pred_label = paddle.argmax(pred_label, -1)[_valid_index].cpu().numpy()
                    pred_label = [args.vocab.id_to_label(str(i)) for i in pred_label]
                    result.append({"sentence":sentence,"label":self.sequence_tag2tag(sentence,pred_label)})
                batch += args.batch_size
        with open(os.path.join(args.save_path,args.task,"model_predicted.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='参照组')
    parser.add_argument('--save_path', type=str, default='./outputs')
    parser.add_argument('--predict_path', type=str, default='./outputs')
    parser.add_argument('--bert_name', type=str, default=r"bert-base-chinese")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--conv_hid_size', type=int, default=96)
    parser.add_argument('--dilation', type=list, default=[1,2,3,4])
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args(args=[])

    dataset, ori_data = data_loader.load_real_bert(args)
    data_loader = paddle.io.DataLoader(dataset=dataset
                                       ,collate_fn=data_loader.pred_collate_fn
                                       ,batch_size=args.batch_size
                                    )
    model = models.bertCNN(args)
    predictor = Predictor(model,args)
    predictor.load(os.path.join("./outputs",args.task,model.model_name + ".pt"))
    predictor.predcit(data_loader,ori_data)
