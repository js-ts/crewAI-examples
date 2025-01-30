from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional

class InputSchema(BaseModel):
    func_name: str
    input_type: Optional[str] = None
    func_input_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = None
