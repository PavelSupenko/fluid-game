#if DebugLog
using MeltIt.Services.Cheats;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
using JetBrains.Annotations;
using Newtonsoft.Json;
using VContainer.Unity;

namespace MeltIt.Features.Cheats
{
    public class CheatService : ICheatService, IDisposable, IInitializable
    {
        private const string PlayerPrefsKey = "CheatsData";
        
        [CanBeNull] public event Action<CheatActionInfo> ActionRegistered;
        [CanBeNull] public event Action<CheatActionInfo> ActionUnregistered;
        [CanBeNull] public event Action<CheatPropertyInfo> PropertyRegistered;
        [CanBeNull] public event Action<CheatPropertyInfo> PropertyUnregistered;

        public List<CheatActionInfo> Actions { get; } = new();
        public List<CheatPropertyInfo> Properties { get; } = new();
        public BoolCheatProperty TestDevice => _cheatModel.General.TestDevice;

        private readonly List<(string Command, Action Action)> _waitingPropertiesToAttach = new();
        private readonly List<(string Command, Action Action)> _waitingActionsToAttach = new();
        private readonly CheatModel _cheatModel;

        public CheatService()
        {
            _cheatModel = Load();
        }

        public void Initialize()
        {
            TestDevice.ValueChanged += OnPropertyValueChanged;
        }

        public void Dispose()
        {
            TestDevice.ValueChanged -= OnPropertyValueChanged;
            Save();
        }

        public void RegisterAction(string commandName, Action action, string category = null)
        {
            CheatActionInfo actionInfo = new CheatActionInfo
            {
                Name = commandName,
                Actions = new List<Action>() { action },
                Category = category
            };
            Actions.Add(actionInfo);
            ActionRegistered?.Invoke(actionInfo);
            
            var waitingActions = _waitingActionsToAttach.Where(item => item.Command == commandName).ToList();
            foreach (var waitingAction in waitingActions)
            {
                actionInfo.Actions.Add(waitingAction.Action);
                _waitingActionsToAttach.Remove(waitingAction);
            }
        }

        public void UnregisterAction(Action action)
        {
            var actionsToRemove = Actions.Where(info => info.Actions.Contains(action)).ToList();
            foreach (CheatActionInfo actionInfo in actionsToRemove)
            {
                Actions.Remove(actionInfo);
                ActionUnregistered?.Invoke(actionInfo);
            }
        }

        public void UnregisterAction(string commandName)
        {
            var actionsToRemove = Actions.Where(info => info.Name == commandName).ToList();
            foreach (CheatActionInfo actionInfo in actionsToRemove)
            {
                Actions.Remove(actionInfo);
                ActionUnregistered?.Invoke(actionInfo);
            }
        }

        public void AttachAction(string commandName, Action action)
        {
            var actionToAttach = Actions.FirstOrDefault(info => info.Name == commandName);
            if (actionToAttach == null)
            {
                _waitingActionsToAttach.Add((commandName, action));
                return;
            }
            
            actionToAttach.Actions.Add(action);
        }
        
        public void DetachAction(string commandName, Action action)
        {
            var actionToDetach = Actions.FirstOrDefault(info => info.Name == commandName);
            actionToDetach?.Actions.Remove(action);
        }

        public void CreateBindProperty(object binder, string name, Func<int> getter, Action<int> setter,
            string category = null, int? min = null, int? max = null)
        {
            IntCheatProperty property = new IntCheatProperty(getter(), switchable: false, min, max);
            RegisterAndBindProperty(binder, name, property, getter, setter, category);
        }

        public void CreateBindProperty(object binder, string name, Func<float> getter, Action<float> setter, 
            string category = null, float? min = null, float? max = null)
        {
            FloatCheatProperty property = new FloatCheatProperty(getter(), switchable: false, min, max);
            RegisterAndBindProperty(binder, name, property, getter, setter, category);
        }

        public void CreateBindProperty(object binder, string name, Func<string> getter, Action<string> setter,
            string category = null)
        {
            StringCheatProperty property = new StringCheatProperty(getter(), switchable: false);
            RegisterAndBindProperty(binder, name, property, getter, setter, category);
        }

        public void CreateBindProperty(object binder, string name, Func<bool> getter, Action<bool> setter,
            string category = null)
        {
            BoolCheatProperty property = new BoolCheatProperty(getter(), switchable: false);
            RegisterAndBindProperty(binder, name, property, getter, setter, category);
        }

        public void CreateBindProperty<TEnum>(object binder, string name, Func<TEnum> getter, Action<TEnum> setter,
            string category = null) where TEnum : struct, Enum
        {
            Type enumType = typeof(TEnum);
            string currentValue = getter().ToString();
            EnumCheatProperty property = new EnumCheatProperty(enumType, currentValue, switchable: false);
            RegisterAndBindProperty(binder, name, property, 
                () => getter().ToString(),
                value => setter((TEnum)Enum.Parse(enumType, value)), 
                category);
        }

        public void CreateBindProperty<TArray>(object binder, string name, TArray[] array, 
            Func<TArray> getter, Action<TArray> setter, string category = null)
        {
            string currentValue = getter().ToString();
            string[] stringArray = array.Select(item => item.ToString()).ToArray();
            ArrayCheatProperty property = new ArrayCheatProperty(stringArray, currentValue, switchable: false);
            RegisterAndBindProperty(binder, name, property,
                () => getter().ToString(),
                value =>
                {
                    int stringValueIndex = Array.IndexOf(stringArray, value);
                    if (stringValueIndex >= 0 && stringValueIndex < array.Length)
                        setter(array[stringValueIndex]);
                }, 
                category);
        }
        
        public void CreateBindProperty(object binder, string name, 
            Func<TimeSpan> getter, Action<TimeSpan> setter, string category = null)
        {
            TimeSpanCheatProperty property = new TimeSpanCheatProperty(getter(), switchable: false);
            RegisterAndBindProperty(binder, name, property, getter, setter, category);
        }

        private void RegisterAndBindProperty<T>(object binder, string name, ACheatProperty<T> property, 
            Func<T> getter, Action<T> setter, string category = null)
        {
            property.Bind(binder, getter, setter);
            CheatPropertyInfo propertyInfo = new CheatPropertyInfo
            {
                Name = name,
                Property = property,
                Category = category
            };
            Properties.Add(propertyInfo);
            PropertyRegistered?.Invoke(propertyInfo);
            property.ValueChanged += OnPropertyValueChanged;
            
            var waitingProperties = _waitingPropertiesToAttach.Where(item => item.Command == name).ToList();
            foreach (var waitingProperty in waitingProperties)
            {
                property.ValueChanged += waitingProperty.Action;
                _waitingPropertiesToAttach.Remove(waitingProperty);
            }
        }

        public void UnregisterAllProperties(object binder)
        {
            var propertiesToRemove = Properties.ToList();
            foreach (CheatPropertyInfo propertyInfo in propertiesToRemove)
            {
                propertyInfo.Property.Unbind(binder);
                if (propertyInfo.Property.BindingsCount <= 0)
                {
                    propertyInfo.Property.ValueChanged -= OnPropertyValueChanged;
                    PropertyUnregistered?.Invoke(propertyInfo);
                    Properties.Remove(propertyInfo);
                }
            }
        }

        public void UnregisterProperty(object binder, string name)
        {
            var propertiesToRemove = Properties.Where(info => info.Name == name).ToList();
            foreach (CheatPropertyInfo propertyInfo in propertiesToRemove)
            {
                propertyInfo.Property.Unbind(binder);
                if (propertyInfo.Property.BindingsCount <= 0)
                {
                    propertyInfo.Property.ValueChanged -= OnPropertyValueChanged;
                    PropertyUnregistered?.Invoke(propertyInfo);
                    Properties.Remove(propertyInfo);
                }
            }
        }
        
        public void AttachPropertyCallback(string name, Action action)
        {
            var property = Properties.FirstOrDefault(info => info.Name == name);
            if (property == null)
            {
                _waitingPropertiesToAttach.Add((name, action));
                return;
            }
                
            property.Property.ValueChanged += action;
        }
        
        public void DetachPropertyCallback(string name, Action action)
        {
            var property = Properties.FirstOrDefault(info => info.Name == name);
            
            if (property != null)
                property.Property.ValueChanged -= action;
        }

        private void OnPropertyValueChanged() => 
            Save();

        private void Save()
        {
            string json = JsonConvert.SerializeObject(_cheatModel);
            PlayerPrefs.SetString(PlayerPrefsKey, json);
            PlayerPrefs.Save();
        }

        private CheatModel Load()
        {
            CheatModel cached;
            try { cached = JsonConvert.DeserializeObject<CheatModel>(PlayerPrefs.GetString(PlayerPrefsKey, null)); }
            catch { cached = null; }
            
            return cached ?? new CheatModel();
        }
    }
}
#endif