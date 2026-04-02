using MeltIt.Features.Cheats;
using VContainer.Unity;
using VContainer;

namespace MeltIt.Features.Inject
{
    public class GameLifetimeScope : LifetimeScope
    {
        protected override void Configure(IContainerBuilder builder)
        {
#if DebugLog
            builder.Register<CheatService>(Lifetime.Singleton).AsImplementedInterfaces();
#else
            builder.Register<MockCheatService>(Lifetime.Singleton).AsImplementedInterfaces();
#endif
        }
    }
}