#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <deque>
#include <stdexcept> 
#include <cstdint>
#include <memory>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <pybind11/pybind11.h>
class Profiler
{
private:
  std::unordered_map<std::string, int> callCounts;
  const int threshold = 5;

public:
  void count(const std::string &funcName)
  {
    int &counter = callCounts[funcName];
    counter++;
  }
  void report() const
  {
    for (const auto &pair : callCounts)
    {
      std::cout << pair.first << ": " << pair.second << "\n";
    }
  }
};
Profiler profiler;

#define PROFILE_COUNT profiler.count(__func__);
enum class ExprType {
	Value,    
	Operator
};
struct Expression {
	ExprType type;
	std::string value; 
	std::vector<Expression> operands;
};
enum class TokenType
{
  Number,
  Plus,
  Minus,
  Star,
  Slash,
  LParen,
  RParen,
  End,
  Invalid
};
struct Token
{
  TokenType Type;
  std::string Value;
};
//Test function for profiling
void foo()
{
  PROFILE_COUNT;
}

class Tokenizer{

  public:
  // Input string
  std::string str;
  std::vector<Token> tokens;
  Tokenizer(std::string str): str(str){}
// Prints output of the tokenizer
  void printTokOut (){
    std::cout << "--- Tokenized Output ---\n";
  for (const Token& i : tokens)
   {
    std::cout << "Type: " << static_cast<int>(i.Type) << " Value: " << i.Value << "\n";
  }
  std::cout << "------------------------\n";

}
// Tokenizing
void tokenize()
{
  PROFILE_COUNT;
  std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
  str.erase(end_pos, str.end());
  std::vector<Token> res;
  size_t i{0};
  while (i < str.length())
  {
    char c = str[i];

    switch (c)
    {
    case '+':
      res.push_back({TokenType::Plus, "+"});
      break;
    case '-':
      res.push_back({TokenType::Minus, "-"});
      break;
    case '*':
      res.push_back({TokenType::Star, "*"});
      break;
    case '/':
      res.push_back({TokenType::Slash, "/"});
      break;
    case '(':
      res.push_back({TokenType::LParen, "("});
      break;

    case ')':
      res.push_back({TokenType::RParen, ")"});
      break;

    default:

      if (std::isdigit(c))
      {
        std::string num;
        while (i < str.length() && std::isdigit(str[i]))
        {
          num += str[i++];
        }
        res.push_back({TokenType::Number, num});
        continue;
      }
      else
      {
        res.push_back({TokenType::Invalid, std::string(1, c)});
      }
      break;
    }
    i++;
  }
  res.push_back({TokenType::End, ""});

  tokens = res;
}
};

class Parser {
private:
	std::deque<Token> tokens;
	Token next() {
		Token token = tokens.front();
		tokens.pop_front();
		return token;

	}
	Token peek() {
		return tokens.front();
	}
	bool match(std::string expected) {
		if (!tokens.empty() && peek().Value == expected) {
			next();
			return true;
		}
		return false;
	}
public:
	Parser(const std::vector<Token>& tokenVec) : tokens(tokenVec.begin(), tokenVec.end()) {}

	Expression parseExpression() {
        PROFILE_COUNT; 
		Expression left = parseTerm();
		while (!tokens.empty() && (peek().Value == "+" || peek().Value == "-")) {
			Token op = next();
			Expression right = parseTerm();
			Expression newExpr;
			newExpr.type = ExprType::Operator; 
			newExpr.value = op.Value;
			newExpr.operands = { left , right };
			left = newExpr;
		}
		return left;
	}

	Expression parseTerm() {
        PROFILE_COUNT; 
		Expression left = parseFactor();
		while (!tokens.empty() && (peek().Value == "*" || peek().Value == "/")) {
			Token op = next();
			Expression right = parseFactor();
			Expression newExpr;
            newExpr.type = ExprType::Operator; 
			newExpr.value = op.Value;
			newExpr.operands = { left , right };
			left = newExpr;
		}
		return left;
	}

	Expression parseFactor() {
        PROFILE_COUNT; 
		Token t = next();
		if (t.Value == "(") {
			Expression inner = parseExpression();
			if (next().Value != ")") {
				throw std::runtime_error("Expected a  ')'");
			}
			return inner;
		}
		if (t.Type == TokenType::Number) {
			return Expression{ ExprType::Value,t.Value,{} };
		}
		throw std::runtime_error("Unexpected token in factor");

	}
};



size_t offset = 0; 


/*
class UOp:
  op: Ops
  dtype: dtypes
  src: tuple(UOp)
  arg: None
*/

enum class Ops{
  ADD,
  SUB,
  MUL, 
  DIV,
  EMPTY,
};
enum class Dtypes{
  INT, 
  FLOAT,
  DOUBLE
  
};
std::string to_string(Ops op) {
    switch (op) {
        case Ops::EMPTY: return "Ops.EMPTY";
        case Ops::ADD:   return "Ops.ADD";
        case Ops::SUB:   return "Ops.SUB";
        case Ops::MUL:   return "Ops.MUL";
        case Ops::DIV:   return "Ops.DIV";
    }
    return "Ops.UNKNOWN";
}

std::string to_string(Dtypes dtype) {
    switch (dtype) {
        case Dtypes::INT:    return "dtypes.INT";
        case Dtypes::FLOAT:  return "dtypes.FLOAT";
        case Dtypes::DOUBLE: return "dtypes.DOUBLE";
    }
    return "dtypes.UNKNOWN";
}
class UOp{
  public: 
  Ops op;
  int val;
  Dtypes dtype;
  
  std::vector<std::shared_ptr<UOp>> src;


  UOp(Ops op, int val,Dtypes dtype,   std::vector<std::shared_ptr<UOp>> src = { } ): op(op), val(val),dtype(dtype), src(std::move(src)){}
  
  std::string print_tree(int indent = 0) const {
        std::ostringstream oss;
        std::string pad(indent, ' ');

        oss << pad << "UOp(" << ::to_string(op) << ", " << ::to_string(dtype) << ", " << val;
        if (!src.empty()) {
            oss << ", src=(\n";
            for (size_t i = 0; i < src.size(); ++i) {
                oss << src[i]->print_tree(indent + 2);
                if (i + 1 < src.size()) oss << ",\n";
            }
            oss << "\n" << pad << ")";
        }
        oss << ")";
        return oss.str();
    }
};


std::shared_ptr<UOp> lowerExpression(const Expression& expr){
  if(expr.type== ExprType::Value){
    return std::make_shared<UOp>(Ops::EMPTY, std::stoi(expr.value),Dtypes::INT);
  }
  Ops op;
  op =(expr.value == "+")  ?  Ops::ADD:
      (expr.value == "-") ? Ops::SUB :
      (expr.value == "*") ? Ops::MUL  :
      (expr.value == "/") ? Ops::DIV : throw std::runtime_error("Unknown op: " + expr.value);

  std::vector<std::shared_ptr<UOp>> children;
  for(const auto &child:expr.operands){
    children.push_back(lowerExpression(child));

  }
  return std::make_shared<UOp>(op, NULL,Dtypes::INT,children);
  
}

std::string generateCudaFromUOp(const std::shared_ptr<UOp> &uop, int &tempId, std::ostringstream &code) {
    if (uop->op == Ops::EMPTY) {
        std::string name = "in" + std::to_string(tempId++);
        return name;
    }

    std::string a = generateCudaFromUOp(uop->src[0], tempId, code);
    std::string b = generateCudaFromUOp(uop->src[1], tempId, code);
    std::string out = "tmp" + std::to_string(tempId++);
    std::string opStr = (uop->op == Ops::ADD) ? "+" :
                        (uop->op == Ops::SUB) ? "-" :
                        (uop->op == Ops::MUL) ? "*" :
                        (uop->op == Ops::DIV) ? "/" : "?";
    code << "    float " << out << " = " << a << "[idx] " << opStr << " " << b << "[idx];\n";
    return out;
}

std::string buildCudaKernel(const std::shared_ptr<UOp> &uop) {
    std::ostringstream code;
    code << "__global__ void kernel(float *in0, float *in1, float *out) {\n";
    code << "  int idx = threadIdx.x;\n";

    int tempId = 0;
    std::ostringstream exprCode;
    std::string resultVar = generateCudaFromUOp(uop, tempId, exprCode);

    code << exprCode.str();
    code << "  out[idx] = " << resultVar << ";\n";
    code << "}\n";
    return code.str();
}

int main()
{

    std::string str{"12*12+8"};
    Tokenizer tok(str);
    tok.tokenize();

  std::shared_ptr<UOp> uops = NULL;
  Parser parser(tok.tokens); 

  try {
    Expression result = parser.parseExpression();
    std::cout << "Parsed Expression: " << result.value <<  "\n";
    if (!result.operands.empty()) {
        std::cout << " (" << result.operands[0].value << ", " << result.operands[1].value << ")";
    }
    uops = lowerExpression(result);

    std::cout << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << "Parsing error: " << e.what() << "\n";
  }
  std::cout << uops.get()->print_tree() << std::endl;
  std::string cudaCode = buildCudaKernel(uops);
  std::ofstream("kernel.cu") << cudaCode;
  std::cout << "Generated Kernel!\n";
 //system("nvcc -ptx kernel.cu -o kernel.ptx");

 
}
