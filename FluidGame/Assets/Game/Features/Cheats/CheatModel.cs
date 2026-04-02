using MeltIt.Services.Cheats;

namespace MeltIt.Features.Cheats
{
    public class GeneralCheats
    {
        public BoolCheatProperty TestDevice = new(false);
    }
    
    public class PlayerStateCheats
    {
        public IntCheatProperty LivesRefillTimeMinutes = new(30, switchable: true);
    }

    public class CheatModel
    {
        public GeneralCheats General = new();
        public PlayerStateCheats PlayerState = new();
    }
}