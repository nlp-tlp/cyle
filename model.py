#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm

pi = 3.14159265358979323846


def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


def convert_to_height(x):
    y = (torch.sigmoid(2 * x) - 0.5) * 2 * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class CylinderProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(CylinderProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(
            self.entity_dim + self.relation_dim * 2, self.hidden_dim)
        self.layer0 = nn.Linear(
            self.hidden_dim, self.entity_dim + self.relation_dim * 2)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(
                self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, source_embedding_height,
                r_embedding_axis, r_embedding_arg, r_embedding_height):

        x = torch.cat([source_embedding_axis + r_embedding_axis,
                       source_embedding_arg + r_embedding_arg,
                       source_embedding_height + r_embedding_height], dim=-1)

        for nl in range(1, self.num_layers + 1):
            x = F.silu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg, height = torch.chunk(x, 3, dim=-1)

        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        height_embeddings = convert_to_height(height)

        return axis_embeddings, arg_embeddings, height_embeddings


class CylinderIntersection(nn.Module):
    def __init__(self, dim):
        super(CylinderIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(3 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, axis_embeddings, arg_embeddings, height_embedding):
        # (num_conj, batch_size, 3 * dim)
        all_embeddings = torch.cat(
            [axis_embeddings, arg_embeddings, height_embedding], dim=-1)
        # (num_conj, batch_size, 2 * dim)
        layer1_act = F.silu(self.layer1(all_embeddings))
        # (num_conj, batch_size, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)

        axis_embeddings = torch.sum(attention * axis_embeddings, dim=0)
        arg_embeddings = torch.sum(attention * arg_embeddings, dim=0)
        height_embeddings = torch.sum(attention * height_embedding, dim=0)

        return axis_embeddings, arg_embeddings, height_embeddings


class CylinderNegation(nn.Module):
    def __init__(self):
        super(CylinderNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding, height_embedding):
        indicator_positive = axis_embedding >= 0.
        indicator_negative = axis_embedding < 0.

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding, height_embedding


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, query_name_dict, test_batch_size=1,
                 use_cuda=False, proj_mode=(1600, 2), center_reg=0.2):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = torch.tensor(2.0)
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda(
        ) if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        hidden_dim, num_layers = proj_mode

        if self.geo == 'cylinder':
            self.entity_embedding = nn.Parameter(
                torch.zeros(nentity, self.entity_dim))
            self.entity_height_embedding = nn.Parameter(
                torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_height_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.height_embedding = nn.Parameter(torch.zeros(
                nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.height_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.cylinder_negation = CylinderNegation()
            self.cylinder_proj = CylinderProjection(
                self.entity_dim, hidden_dim, num_layers)
            self.cylinder_intersection = CylinderIntersection(self.entity_dim)

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_embedding = nn.Parameter(torch.zeros(
            nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.arg_embedding = nn.Parameter(torch.zeros(
            nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.angle_scale = AngleScale(self.embedding_range.item())

        self.cen = center_reg
        self.modulus = nn.Parameter(torch.Tensor(
            [0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'cylinder':
            return self.forward_cylinder(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2],
                                            queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4],
                                            queries[:, 5:6]], dim=1)],
                                dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def embed_query_cylinder(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=queries[:, idx])
                height_entity_embedding = torch.index_select(
                    self.entity_height_embedding, dim=0, index=queries[:, idx])

                axis_entity_embedding = self.angle_scale(
                    axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)
                height_entity_embedding = convert_to_height(
                    height_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(
                        axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(
                        axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
                height_embedding = height_entity_embedding
            else:
                axis_embedding, arg_embedding, height_embedding, idx = self.embed_query_cylinder(
                    queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding, height_embedding = \
                        self.cylinder_negation(
                            axis_embedding, arg_embedding, height_embedding)
                # projection
                else:
                    axis_r_embedding = torch.index_select(
                        self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(
                        self.arg_embedding, dim=0, index=queries[:, idx])
                    height_r_embedding = torch.index_select(
                        self.height_embedding, dim=0, index=queries[:, idx])

                    axis_r_embedding = self.angle_scale(
                        axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(
                        arg_r_embedding, self.arg_scale)

                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)
                    height_r_embedding = convert_to_height(height_r_embedding)

                    axis_embedding, arg_embedding, height_embedding = \
                        self.cylinder_proj(axis_embedding, arg_embedding, height_embedding,
                                           axis_r_embedding, arg_r_embedding, height_r_embedding)
                idx += 1
        else:
            # intersection
            axis_embedding_list = []
            arg_embedding_list = []
            height_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, height_embedding, idx = self.embed_query_cylinder(
                    queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)
                height_embedding_list.append(height_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)
            stacked_height_embeddings = torch.stack(height_embedding_list)

            axis_embedding, arg_embedding, height_embedding = \
                self.cylinder_intersection(
                    stacked_axis_embeddings, stacked_arg_embeddings, stacked_height_embeddings)

        return axis_embedding, arg_embedding, height_embedding, idx

    def cal_logit_cylinder(self, entity_embedding, entity_height_embedding,
                           query_axis_embedding, query_arg_embedding, query_height_embedding):
        entity_axis_embedding = self.angle_scale(
            entity_embedding, self.axis_scale)
        entity_axis_embedding = convert_to_axis(entity_axis_embedding)
        entity_height_embedding = convert_to_height(entity_height_embedding)

        low_query = query_axis_embedding - query_arg_embedding
        up_query = query_axis_embedding + query_arg_embedding

        distance2axis = torch.abs(
            1 - torch.cos(entity_axis_embedding - query_axis_embedding))
        distance_base = torch.abs(1 - torch.cos(query_arg_embedding))
        distance_in = torch.min(distance2axis, distance_base)

        distance_out = torch.min(torch.abs(1 - torch.cos(entity_axis_embedding - low_query)),
                                 torch.abs(1 - torch.cos(entity_axis_embedding - up_query)))
        indicator_in = distance2axis < distance_base
        distance_out[indicator_in] = 0.

        distance_height = torch.abs(
            entity_height_embedding - query_height_embedding)

        distance = torch.norm(distance_out, p=1, dim=-1) + \
            self.cen * torch.norm(distance_in, p=1, dim=-1) + \
            torch.norm(distance_height, p=1, dim=-1)

        logit = self.gamma - distance * self.modulus

        return logit

    def forward_cylinder(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_union_idxs = [], []
        all_axis_embeddings, all_arg_embeddings, all_height_embeddings = [], [], []
        all_union_axis_embeddings, all_union_arg_embeddings, all_union_height_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, height_embedding, _ = \
                    self.embed_query_cylinder(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                              self.transform_union_structure(query_structure), 0)
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
                all_union_height_embeddings.append(height_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                axis_embedding, arg_embedding, height_embedding, _ = \
                    self.embed_query_cylinder(
                        batch_queries_dict[query_structure], query_structure, 0)
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)
                all_height_embeddings.append(height_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])
        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(
                all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(
                all_arg_embeddings, dim=0).unsqueeze(1)
            all_height_embeddings = torch.cat(
                all_height_embeddings, dim=0).unsqueeze(1)
            #  assert len(all_height_embeddings[all_height_embeddings < 0.0]) == 0
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(
                all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(
                all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_height_embeddings = torch.cat(
                all_union_height_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_height_embeddings = all_union_height_embeddings.view(
                all_union_height_embeddings.shape[0] // 2, 2, 1, -1)
            #  assert len(all_union_height_embeddings[all_union_height_embeddings < 0.0]) == 0

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular)
                positive_embedding = positive_embedding.unsqueeze(1)
                positive_embedding_height = torch.index_select(self.entity_height_embedding, dim=0,
                                                               index=positive_sample_regular)
                positive_embedding_height = positive_embedding_height.unsqueeze(
                    1)
                positive_logit = self.cal_logit_cylinder(positive_embedding, positive_embedding_height,
                                                         all_axis_embeddings, all_arg_embeddings, all_height_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union)
                positive_embedding = positive_embedding.unsqueeze(
                    1).unsqueeze(1)
                positive_embedding_height = torch.index_select(self.entity_height_embedding, dim=0,
                                                               index=positive_sample_union)
                positive_embedding_height = positive_embedding_height.unsqueeze(
                    1).unsqueeze(1)
                positive_union_logit = self.cal_logit_cylinder(positive_embedding, positive_embedding_height,
                                                               all_union_axis_embeddings, all_union_arg_embeddings,
                                                               all_union_height_embeddings)
                positive_union_logit = torch.max(
                    positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            positive_logit = torch.cat(
                [positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1))
                negative_embedding = negative_embedding.view(
                    batch_size, negative_size, -1)
                negative_embedding_height = torch.index_select(self.entity_height_embedding, dim=0,
                                                               index=negative_sample_regular.view(-1))
                negative_embedding_height = negative_embedding_height.view(
                    batch_size, negative_size, -1)
                negative_logit = self.cal_logit_cylinder(negative_embedding, negative_embedding_height,
                                                         all_axis_embeddings, all_arg_embeddings, all_height_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(
                    self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1))
                negative_embedding = negative_embedding.view(
                    batch_size, 1, negative_size, -1)
                negative_embedding_height = torch.index_select(self.entity_height_embedding, dim=0,
                                                               index=negative_sample_union.view(-1))
                negative_embedding_height = negative_embedding_height.view(
                    batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_cylinder(negative_embedding, negative_embedding_height,
                                                               all_union_axis_embeddings, all_union_arg_embeddings,
                                                               all_union_height_embeddings)
                negative_union_logit = torch.max(
                    negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor(
                    []).to(self.entity_embedding.device)
            negative_logit = torch.cat(
                [negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(
            train_iterator)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        # group queries with same structure
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(
                    batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(
                    None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                if len(argsort) == args.test_batch_size:
                    # achieve the ranking of all entities
                    ranking = ranking.scatter_(
                        1, argsort, model.batch_entity_range)
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(
                        easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    # only take indices that belong to the hard answers
                    cur_ranking = cur_ranking[masks]

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' %
                                 (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum(
                    [log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(
                logs[query_structure])

        return metrics
