//here are the definitions of the CalcTransit and CalcRV structure.
typedef struct {
  int planet;
  int epoch;
  double time;
  double rsky;
  double vsky;} CalcTransit;

typedef struct {
  double time;
  double RV;} CalcRV;
