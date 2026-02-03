from typing import TypedDict, Union, Optional, Any, Annotated

class Movie(TypedDict):
    title: str
    year: int

movie = Movie(title="Inception", year=2010)
print(movie)

def square(value: Union[int, float]) -> Union[int, float]:
    return value * value

def say_hi(name: Optional[str] = None) -> str:
    if name:
        return f"Hi, {name}!"
    return "Hi!"

def print_any(value: Any) -> None:
    print(value)

email = Annotated[str, "This has to be a valid email address"]
print(email.__metadata__)

# state - node - graph - edges
# conditional edges - start - end
# tools(special functions) - toolNode(connects tool output into the state)
# StateGraph(used to build and compile the graph) - Runnable()
# messages: human - system - function - AI - tool