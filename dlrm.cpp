#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <thread>
#include <tuple>

class DLRMImpl : torch::nn::Module{
public:
    int64_t ndevices;
    int64_t output_d;
    int64_t parallel_model_batch_size;
    bool parallel_model_is_not_prepared;
    std::string arch_interaction_op;
    bool arch_interaction_itself;
    bool sync_dense_params;
    float loss_threshold;
    std::string loss_function;
    std::string weighted_pooling;
    bool qr_flag;
    std::string qr_operation;
    int64_t qr_collisions;
    int64_t qr_threshold;
    bool md_flag;
    int64_t md_threshold;
    int64_t n_global_emb;
    int64_t n_local_emb;
    int64_t n_emb_per_rank;
    std::vector<int64_t> local_emb_slice;
    std::vector<int64_t> local_emb_indices;
    torch::nn::ModuleList emb_l{nullptr};
    torch::nn::Sequential bot_l{nullptr}, top_l{nullptr};
    bool quantize_emb;
    int64_t quantize_bits;
    torch::nn::MSELoss loss_fn{nullptr};
    torch::nn::ParameterList v_W_l{nullptr};
    std::vector<torch::nn::Sequential> bot_l_replicas{nullptr}, top_l_replicas{nullptr};

    /*
        ln: 每层神经元数量列表
        sigmoid_layer: 应用sigmoid激活函数的层索引
    */
    torch::nn::Sequential create_mlp(std::vector<int64_t> ln, int64_t sigmoid_layer){
        torch::nn::Sequential layers;
        // layers = register_module("layers", torch::nn::Sequential());
        for(int i=0; i<ln.size()-1;i++){
            int64_t n = ln[i];
            int64_t m = ln[i+1];

            auto linear = torch::nn::Linear(torch::nn::LinearOptions(n, m).bias(true));
            float mean = 0.0;
            float std_dev = sqrt(2.0f/(m+n));
            auto W = torch::randn({m, n}) * std_dev + mean;  // 等价于 numpy.normal

            // 初始化 bias bt: shape [m]
            std_dev = std::sqrt(1.0f / m);
            auto bt = torch::randn({m}) * std_dev + mean;

            linear->weight.set_data(W.clone().set_requires_grad(true));
            linear->bias.set_data(bt.clone().set_requires_grad(true));

            layers->push_back(linear);

            if(i == sigmoid_layer){
                layers->push_back(torch::nn::Sigmoid());
            }else{
                layers->push_back(torch::nn::ReLU());
            }
        }

        return layers;
    }

    std::tuple<torch::nn::ModuleList, std::vector<torch::Tensor>> create_emb(int64_t m, std::vector<int64_t> ln, std::string weighted_pooling=""){
        torch::nn::ModuleList emb_l;
        emb_l = register_module("emb_l", torch::nn::ModuleList());
        std::vector<torch::Tensor> v_W_l;
        for(int i=0; i<ln.size(); i++){
            int64_t n = ln[i];
            torch::nn::EmbeddingBag EE = torch::nn::EmbeddingBag(torch::nn::EmbeddingBagOptions(n,m).mode(torch::kSum).sparse(true));
            float bound = std::sqrt(1.0f / n);
            auto W = torch::empty({n, m}, torch::kFloat32)
                    .uniform_(-bound, bound);
            EE->weight.set_data(W.clone().set_requires_grad(true));
            if(weighted_pooling == ""){
                v_W_l.push_back(torch::Tensor{nullptr});
            }else{
                v_W_l.push_back(torch::ones(n, torch::kFloat32));
            }
            emb_l->push_back(EE);
        } 
        return {emb_l, v_W_l};
    }

    torch::Tensor interact_features(torch::Tensor x, std::vector<torch::Tensor> ly) {
        torch::Tensor R;
        
        if (arch_interaction_op == "dot") {
            // 获取batch大小和维度
            auto batch_size = x.size(0);
            auto d = x.size(1);
            
            // 连接密集特征和稀疏特征
            std::vector<torch::Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            torch::Tensor T = torch::cat(concat_list, 1).view({batch_size, -1, d});
            
            // 执行点积操作
            torch::Tensor T_transpose = T.transpose(1, 2);
            torch::Tensor Z = torch::bmm(T, T_transpose);
            
            // 提取下三角部分
            auto sizes = Z.sizes();
            int64_t ni = sizes[1];
            int64_t nj = sizes[2];
            
            // 创建索引
            int64_t offset = arch_interaction_itself ? 1 : 0;
            std::vector<int64_t> li_vec, lj_vec;
            
            for (int64_t i = 0; i < ni; i++) {
                for (int64_t j = 0; j < i + offset; j++) {
                    li_vec.push_back(i);
                    lj_vec.push_back(j);
                }
            }
            
            auto options = torch::TensorOptions().dtype(torch::kLong).device(Z.device());
            torch::Tensor li = torch::tensor(li_vec, options);
            torch::Tensor lj = torch::tensor(lj_vec, options);
            
            // 提取特定的索引位置
            torch::Tensor Zflat = Z.index({torch::indexing::Slice(), li, lj});
            
            // 连接密集特征和交互特征
            R = torch::cat({x, Zflat}, 1);
        } 
        else if (arch_interaction_op == "cat") {
            // 简单连接所有特征
            std::vector<torch::Tensor> concat_list = {x};
            concat_list.insert(concat_list.end(), ly.begin(), ly.end());
            R = torch::cat(concat_list, 1);
        } 
        else {
            throw std::runtime_error("ERROR: --arch-interaction-op=" + arch_interaction_op + " is not supported");
        }
        
        return R;
    }

    // 复制模块到多个设备
    std::vector<torch::nn::Sequential> replicate_modules(
        torch::nn::Sequential& Sequential, 
        const std::vector<int64_t>& device_ids) {
        
        std::vector<torch::nn::Sequential> replicas;
        for (auto device_id : device_ids) {
            torch::Device device(torch::kCUDA, device_id);
            auto replica = Sequential->clone();
            torch::nn::Sequential replica_s = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(replica);
            replica_s->to(device);
            replicas.push_back(replica_s);
        }
        return replicas;
    }
    // 在多个模块上并行应用输入
    std::vector<torch::Tensor> parallel_apply_modules(
        std::vector<torch::nn::Sequential>& Sequential,
        std::vector<torch::Tensor>& inputs) {
        
        std::vector<torch::Tensor> outputs(Sequential.size());
        std::vector<std::thread> threads;
        
        for (size_t i = 0; i < Sequential.size(); i++) {
            threads.emplace_back([&, i]() {
                outputs[i] = Sequential[i]->forward(inputs[i]);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return outputs;
    }
    // 将张量分散到多个设备
    std::vector<torch::Tensor> scatter_tensors(
        const torch::Tensor& tensor, 
        const std::vector<int64_t>& device_ids,
        int64_t dim = 0) {
        
        std::vector<torch::Tensor> outputs;
        int64_t chunk_size = tensor.size(dim) / device_ids.size();
        int64_t remainder = tensor.size(dim) % device_ids.size();
        
        int64_t start = 0;
        for (size_t i = 0; i < device_ids.size(); i++) {
            int64_t end = start + chunk_size + (i < remainder ? 1 : 0);
            torch::Device device(torch::kCUDA, device_ids[i]);
            
            torch::Tensor slice = tensor.slice(dim, start, end).to(device);
            outputs.push_back(slice);
            
            start = end;
        }
        
        return outputs;
    }

    // 收集多个设备上的张量
    torch::Tensor gather_tensors(
        const std::vector<torch::Tensor>& tensors, 
        int64_t dim = 0) {
        
        std::vector<torch::Tensor> aligned_tensors;
        torch::Device target_device = tensors[0].device();
        
        for (const auto& tensor : tensors) {
            aligned_tensors.push_back(tensor.to(target_device));
        }
        
        return torch::cat(aligned_tensors, dim);
    }

    // 应用嵌入操作
    std::vector<torch::Tensor> apply_emb(
        const torch::Tensor& lS_o,
        const std::vector<torch::Tensor>& lS_i,
        torch::nn::ModuleList& emb_l,
        torch::nn::ParameterList& v_W_l) {
        
        std::vector<torch::Tensor> ly;
        
        for (size_t k = 0; k < lS_i.size(); k++) {
            torch::Tensor sparse_index_group_batch = lS_i[k];
            torch::Tensor sparse_offset_group_batch = lS_o[k];
            
            // 确保数据在相同的设备上
            auto device = emb_l[k]->parameters()[0].device();
            sparse_index_group_batch = sparse_index_group_batch.to(device);
            sparse_offset_group_batch = sparse_offset_group_batch.to(device);
            
            torch::Tensor per_sample_weights;
            if (v_W_l[k].defined()) {
                per_sample_weights = v_W_l[k].index_select(0, sparse_index_group_batch).to(device);
            }
            
            // 使用正确的方式获取嵌入表
            auto E = emb_l[k];
            auto emb_bag = std::dynamic_pointer_cast<torch::nn::EmbeddingBagImpl>(E);
            
            if (!emb_bag) {
                // 如果不是EmbeddingBag类型，可能是普通Embedding
                auto emb = std::dynamic_pointer_cast<torch::nn::EmbeddingImpl>(E);
                if (!emb) {
                    throw std::runtime_error("错误: 嵌入层类型不支持");
                }
                
                torch::nn::functional::EmbeddingBagFuncOptions options = 
                    torch::nn::functional::EmbeddingBagFuncOptions()
                        .offsets(sparse_offset_group_batch)
                        .mode(torch::kSum)
                        .scale_grad_by_freq(false)
                        .sparse(true);
                        
                if (per_sample_weights.defined()) {
                    options.per_sample_weights(per_sample_weights);
                }
                
                torch::Tensor V = torch::nn::functional::embedding_bag(
                    sparse_index_group_batch,
                    emb->weight,
                    options);
                    
                ly.push_back(V);
            } else {
                // 直接使用EmbeddingBag的forward函数
                torch::Tensor V;
                if (per_sample_weights.defined()) {
                    V = emb_bag->forward(sparse_index_group_batch, sparse_offset_group_batch, per_sample_weights);
                } else {
                    V = emb_bag->forward(sparse_index_group_batch, sparse_offset_group_batch);
                }
                ly.push_back(V);
            }
        }
        
        return ly;
    }

    torch::Tensor parallel_forward(
        torch::Tensor dense_x,
        torch::Tensor lS_o,
        std::vector<torch::Tensor> lS_i) {
        
        // 正确设备转移
        torch::Device d1(torch::kCUDA, 0);
        dense_x = dense_x.to(d1);  // 注意这里保存结果
        
        // 准备模型
        int64_t batch_size = dense_x.size(0);
        int64_t ndevices = std::min({this->ndevices, batch_size, static_cast<int64_t>(emb_l->size())});
        std::vector<int64_t> device_ids(ndevices);
        for (int64_t i = 0; i < ndevices; i++) {
            device_ids[i] = i;
        }
        
        // 判断是否需要重新分配模型
        if (parallel_model_batch_size != batch_size) {
            parallel_model_is_not_prepared = true;
        }
        
        // 复制MLP实现数据并行
        if (parallel_model_is_not_prepared || sync_dense_params) {
            bot_l_replicas = replicate_modules(bot_l, device_ids);
            top_l_replicas = replicate_modules(top_l, device_ids);
            parallel_model_batch_size = batch_size;
        }
        
        // 分布嵌入表实现模型并行
        if (parallel_model_is_not_prepared) {
            std::vector<std::shared_ptr<torch::nn::Module>> t_list;
            std::vector<torch::Tensor> w_list;
            
            for (size_t k = 0; k < emb_l->size(); k++) {
                torch::Device d(torch::kCUDA, k % ndevices);
                auto module = emb_l[k];  // shared_ptr<Module>
                module->to(d);
                t_list.push_back(module); 
                
                if (weighted_pooling == "learned") {
                    w_list.push_back(v_W_l[k].to(d).clone().detach().requires_grad_(true));
                } 
                else if (weighted_pooling == "fixed") {
                    w_list.push_back(v_W_l[k].to(d));
                } 
                else {
                    w_list.push_back(torch::Tensor());
                }
            }
            
            emb_l = torch::nn::ModuleList();  // 清空或新建

            for (auto& m : t_list) {
                emb_l->push_back(m);  // m 是 shared_ptr<Module>
            }

            v_W_l = torch::nn::ParameterList();
            for(auto& m:w_list){
                v_W_l->append(m);
            }
            parallel_model_is_not_prepared = false;
        }
        
        // 准备输入
        // 分散密集特征 (数据并行)
        std::vector<torch::Tensor> dense_x_list = scatter_tensors(dense_x, device_ids);
        
        // 分布稀疏特征 (模型并行)
        if (emb_l->size() != lS_o.sizes()[0] || emb_l->size() != lS_i.size()) {
            throw std::runtime_error("ERROR: corrupted model input detected in parallel_forward call");
        }
        
        std::vector<torch::Tensor> t_list, i_list;
        for (size_t k = 0; k < emb_l->size(); k++) {
            torch::Device d(torch::kCUDA, k % ndevices);
            t_list.push_back(lS_o[k].to(d));
            i_list.push_back(lS_i[k].to(d));
        }
        lS_o = torch::stack(t_list);
        lS_i = i_list;
        
        // 并行计算
        // 底层MLP (数据并行)
        std::vector<torch::Tensor> x = parallel_apply_modules(bot_l_replicas, dense_x_list);
        
        // 嵌入层
        std::vector<torch::Tensor> ly = apply_emb(lS_o, lS_i, emb_l, v_W_l);
        
        // 蝴蝶重组
        if (emb_l->size() != ly.size()) {
            throw std::runtime_error("ERROR: corrupted intermediate result in parallel_forward call");
        }
        
        std::vector<std::vector<torch::Tensor>> t_list_2d;
        for (size_t k = 0; k < emb_l->size(); k++) {
            torch::Device d(torch::kCUDA, k % ndevices);
            std::vector<torch::Tensor> scattered = scatter_tensors(ly[k], device_ids);
            t_list_2d.push_back(scattered);
        }
        
        // 调整列表顺序按设备
        std::vector<std::vector<torch::Tensor>> ly_per_device(ndevices);
        for (size_t i = 0; i < ndevices; i++) {
            ly_per_device[i].resize(emb_l->size());
            for (size_t j = 0; j < emb_l->size(); j++) {
                ly_per_device[i][j] = t_list_2d[j][i];
            }
        }
        
        // 特征交互
        std::vector<torch::Tensor> z;
        for (int64_t k = 0; k < ndevices; k++) {
            z.push_back(interact_features(x[k], ly_per_device[k]));
        }
        
        // 顶层MLP
        std::vector<torch::Tensor> p = parallel_apply_modules(top_l_replicas, z);
        
        // 收集分布式结果
        torch::Tensor p0 = gather_tensors(p, 0);
        
        // 裁剪输出
        torch::Tensor z0;
        if (0.0 < loss_threshold && loss_threshold < 1.0) {
            z0 = torch::clamp(p0, loss_threshold, 1.0 - loss_threshold);
        } else {
            z0 = p0;
        }
        
        return z0;
    }

    torch::Tensor forward(torch::Tensor dense_x,
        torch::Tensor lS_o,
        std::vector<torch::Tensor> lS_i){
            return parallel_forward(dense_x, lS_o, lS_i);
        }
    
    // 构造函数
    DLRMImpl(
        int64_t m_spa,
        const std::vector<int64_t>& ln_emb,
        const std::vector<int64_t>& ln_bot,
        const std::vector<int64_t>& ln_top,
        const std::string& arch_interaction_op,
        bool arch_interaction_itself,
        int64_t sigmoid_bot,
        int64_t sigmoid_top,
        bool sync_dense_params,
        float loss_threshold,
        int64_t ndevices,
        bool qr_flag,
        const std::string& qr_operation,
        int64_t qr_collisions,
        int64_t qr_threshold,
        bool md_flag,
        int64_t md_threshold,
        const std::string& weighted_pooling,
        const std::string& loss_function) {
        
        if (m_spa != 0 && !ln_emb.empty() && !ln_bot.empty() && 
            !ln_top.empty() && !arch_interaction_op.empty()) {
            
            // 保存参数
            this->ndevices = ndevices;
            this->output_d = 0;
            this->parallel_model_batch_size = -1;
            this->parallel_model_is_not_prepared = true;
            this->arch_interaction_op = arch_interaction_op;
            this->arch_interaction_itself = arch_interaction_itself;
            this->sync_dense_params = sync_dense_params;
            this->loss_threshold = loss_threshold;
            this->loss_function = loss_function;
            
            if (weighted_pooling != "" && weighted_pooling != "fixed") {
                this->weighted_pooling = "learned";
            } else {
                this->weighted_pooling = weighted_pooling;
            }
            
            // 创建QR嵌入相关变量
            this->qr_flag = qr_flag;
            if (this->qr_flag) {
                this->qr_collisions = qr_collisions;
                this->qr_operation = qr_operation;
                this->qr_threshold = qr_threshold;
            }
            
            // 创建MD嵌入相关变量
            this->md_flag = md_flag;
            if (this->md_flag) {
                this->md_threshold = md_threshold;
            }
            
            this->v_W_l = register_module("v_W_l", torch::nn::ParameterList());
            
            // 创建操作符 - 无论ndevices如何都初始化嵌入层
            std::vector<torch::Tensor> w_list;
            std::tie(this->emb_l, w_list) = this->create_emb(m_spa, ln_emb, weighted_pooling);
            
            if (this->weighted_pooling == "learned") {
                for (auto& w : w_list) {
                    this->v_W_l->append(register_parameter("v_W_" + std::to_string(v_W_l->size()), w));
                }
            } else {
                for (auto& w : w_list) {
                    this->v_W_l->append(w);
                }
            }
            
            this->bot_l = register_module("bot_l", this->create_mlp(ln_bot, sigmoid_bot));
            this->top_l = register_module("top_l", this->create_mlp(ln_top, sigmoid_top));
            
            // 量化
            this->quantize_emb = false;
            this->quantize_bits = 32;
            
            // 指定损失函数
            this->loss_fn = register_module("loss_fn", torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
        }
    }
};
TORCH_MODULE(DLRM);


int main() {
    // 模型参数
    std::vector<int64_t> ln_emb = {10, 3, 2, 1};  // 嵌入表大小
    std::vector<int64_t> ln_bot = {4, 3, 2};  // 底层MLP结构
    int64_t num_fea = ln_emb.size() + 1;
    int64_t m_den_out = ln_bot[ln_bot.size() - 1]; 
    int64_t num_int = (num_fea * (num_fea - 1)) / 2 + m_den_out;
    std::vector<int64_t> ln_top = {num_int, 4, 2, 1};
    int64_t mini_batch_size = 11;            // 批次大小
    int64_t num_indices_per_lookup = 2;      // 每个样本每个特征的索引数量
    bool fixed_indices = false;              // 是否使用固定数量的索引
    std::string arch_interaction_op = "dot";
    bool arch_interaction_itself = false;
    int64_t sigmoid_bot = -1;
    int64_t sigmoid_top = ln_top.size() - 2;
    bool sync_dense_params = true;
    float loss_threshold = 0.0;
    int64_t ngpus = 1;
    int64_t ndevices = std::min({ngpus, mini_batch_size, num_fea - 1});
    bool qr_flag = false;
    std::string qr_operation = "mult";
    int64_t qr_collisions = 4;
    int64_t qr_threshold = 200;
    bool md_flag = false;
    int64_t md_threshold = 200;
    std::string weighted_pooling = "";
    std::string loss_function = "mse";


    
    // 设置随机种子
    torch::manual_seed(42);
    
    // 1. 生成密集特征 X: [mini_batch_size, ln_bot[0]]
    torch::Tensor dense_x = torch::rand({mini_batch_size, ln_bot[0]}, torch::kFloat32);
    
    // 可选：移动到GPU（如果可用）
    if (torch::cuda::is_available()) {
        dense_x = dense_x.to(torch::kCUDA);
        std::cout << "Using CUDA" << std::endl;
    }
    
    std::cout << "dense_x shape: " << dense_x.sizes() << std::endl;
    std::cout << dense_x << std::endl << std::endl;
    
    // 2. 生成稀疏特征偏移量和索引值
    std::vector<int64_t> offsets_data;
    std::vector<std::vector<int64_t>> indices_data(ln_emb.size());
    
    // 为每个样本确定特征值数量
    std::vector<std::vector<int64_t>> indices_per_sample(ln_emb.size(), std::vector<int64_t>(mini_batch_size));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            if (fixed_indices) {
                indices_per_sample[f][i] = num_indices_per_lookup;
            } else {
                indices_per_sample[f][i] = torch::randint(1, num_indices_per_lookup + 1, {1}).item<int64_t>();
            }
        }
    }
    
    // 计算每个特征的累积偏移量
    std::vector<std::vector<int64_t>> offsets_per_feature(ln_emb.size(), std::vector<int64_t>(mini_batch_size + 1, 0));
    
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i < mini_batch_size; i++) {
            offsets_per_feature[f][i + 1] = offsets_per_feature[f][i] + indices_per_sample[f][i];
        }
        
        // 生成随机索引
        indices_data[f].resize(offsets_per_feature[f][mini_batch_size]);
        for (int64_t i = 0; i < offsets_per_feature[f][mini_batch_size]; i++) {
            indices_data[f][i] = torch::randint(0, ln_emb[f], {1}).item<int64_t>();
        }
    }
    
    // 创建lS_o张量 (注意这里是整个张量，不是列表)
    torch::Tensor lS_o = torch::zeros({static_cast<int64_t>(ln_emb.size()), mini_batch_size + 1}, torch::kInt64);
    for (size_t f = 0; f < ln_emb.size(); f++) {
        for (int64_t i = 0; i <= mini_batch_size; i++) {
            lS_o[f][i] = offsets_per_feature[f][i];
        }
    }
    
    // 创建lS_i列表（注意这是一个包含张量的向量）
    std::vector<torch::Tensor> lS_i;
    for (size_t f = 0; f < ln_emb.size(); f++) {
        torch::Tensor indices_tensor = torch::zeros({static_cast<int64_t>(indices_data[f].size())}, torch::kInt64);
        for (size_t i = 0; i < indices_data[f].size(); i++) {
            indices_tensor[i] = indices_data[f][i];
        }
        lS_i.push_back(indices_tensor);
    }
    
    // 由于lS_o的形状与示例不完全匹配（示例是[3, 10]而不是[3, 11]），我们修剪最后一列
    lS_o = lS_o.index({"...", torch::indexing::Slice(0, mini_batch_size)});
    
    // 3. 打印生成的数据
    std::cout << "lS_o shape: " << lS_o.sizes() << std::endl;
    std::cout << lS_o << std::endl << std::endl;
    
    std::cout << "lS_i (list of tensors):" << std::endl;
    for (size_t f = 0; f < lS_i.size(); f++) {
        std::cout << "  - Feature " << f << " shape: " << lS_i[f].sizes() << std::endl;
        std::cout << "    " << lS_i[f] << std::endl;
    }
    
    auto dlrm_model = DLRM(m_den_out, ln_emb, ln_bot, ln_top, arch_interaction_op, arch_interaction_itself, sigmoid_bot, sigmoid_top, sync_dense_params,
                    loss_threshold, ndevices, qr_flag, qr_operation, qr_collisions, qr_threshold, md_flag, md_threshold, weighted_pooling, loss_function);
    torch::Tensor result = dlrm_model->forward(dense_x, lS_o, lS_i);
    std::cout<<"result shape: "<<result.sizes()<<std::endl;
    // 4. 调用DLRM前向传播（示例代码）
    /*
    auto dlrm = DLRM_Net(...);
    torch::Tensor output = dlrm->forward(dense_x, lS_o, lS_i);
    */
    
    return 0;
}