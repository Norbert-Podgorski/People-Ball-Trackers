import sys
from pathlib import Path

def add_path_to_python_path(path: Path):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

def remove_path_from_python_path(path: Path):
    path_text = str(path)
    sys.path.remove(path_text)


THIS_DIR = Path(__file__).parent
CODE_PATH = THIS_DIR / "repository/DaSiamRPN/code"
add_path_to_python_path(CODE_PATH)
OLD_UTILS = sys.modules.pop("utils", None)

from src.dasiam_rpn_net.repository.DaSiamRPN.code import net
from src.dasiam_rpn_net.repository.DaSiamRPN.code import utils
from src.dasiam_rpn_net.repository.DaSiamRPN.code import run_SiamRPN

remove_path_from_python_path(CODE_PATH)
sys.modules.pop("utils")
if OLD_UTILS:
    sys.modules["utils"] = OLD_UTILS
