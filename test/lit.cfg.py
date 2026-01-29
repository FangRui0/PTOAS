import os
import lit.formats
from lit.llvm import llvm_config
config.name = 'PTO'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.pto_obj_root, 'test')
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
llvm_config.add_tool_substitutions(["ptoas"], [config.pto_tools_dir])
