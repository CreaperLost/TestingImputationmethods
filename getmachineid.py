import interpretableai
interpretableai.install_julia()
interpretableai.install_system_image()
from interpretableai import iai
print(iai.get_machine_id())