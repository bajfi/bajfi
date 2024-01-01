#include "DrawTool.h"

int main() {
    DrawMethodA impA = DrawMethodA();
    DrawMethodB impB = DrawMethodB();

    CircleDrawTool cir = CircleDrawTool(impA);
    TriangleDrawTool tri = TriangleDrawTool(&impB);

    cir.Draw();
    tri.Draw();

    cir.setMethod(impB);
    cir.Draw();

    return 0;
}
