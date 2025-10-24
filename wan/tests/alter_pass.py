# 测试脚本

def create_alternating_pass(*block_lists, grad_checkpoint):
    max_length = max(len(block_list) for block_list in block_lists) if block_lists else 0
    call_sequence = []
    for i in range(max_length):
        for block_list in block_lists:
            if i < len(block_list):
                call_sequence.append(block_list[i])
    
    def alter_pass(x, c):
        for transformer_block in call_sequence:
            x = grad_checkpoint(transformer_block, x, c)
        return x
    
    return alter_pass

def test_alternating_pass():
    # 定义一个空白的grad_checkpoint函数
    def grad_checkpoint(fn, x, c):
        # 这是一个简单的包装器，只是直接调用函数
        return fn(x, c)
    
    # 测试用的block函数，记录调用顺序
    call_order = []
    
    def block(x, c):
        call_order.append("block")
        return x + 1
    
    def block1(x, c):
        call_order.append("block1")
        return x + 2
    
    def block2(x, c):
        call_order.append("block2")
        return x + 3
    
    # 定义block列表
    block_list_a = [block, block, block, block, block, block]
    block_list_b = [block1, block1]
    block_list_c = [block2]
    
    # 创建alternating pass函数
    alter_pass = create_alternating_pass(block_list_a, block_list_b, block_list_c, grad_checkpoint=grad_checkpoint)
    
    # 测试
    result = alter_pass(0, 10)
    
    # 打印结果和调用顺序
    print(f"调用顺序: {' -> '.join(call_order)}")
    print(f"最终结果: {result}")

if __name__ == "__main__":
    test_alternating_pass()