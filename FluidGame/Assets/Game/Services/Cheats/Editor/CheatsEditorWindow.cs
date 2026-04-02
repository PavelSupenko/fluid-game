using MeltIt.Services.Editor;
using System.Collections.Generic;
using Cysharp.Threading.Tasks;
using System.Threading.Tasks;
using System.Globalization;
using VContainer.Unity;
using System.Linq;
using UnityEditor;
using UnityEngine;
using VContainer;
using System;

namespace MeltIt.Services.Cheats.Editor
{
    public class CheatsEditorWindow : EditorWindow
    {
        private const string DefaultCategoryName = "Context Actions and Variables";
        
        private readonly Dictionary<string, List<object>> _categorizedItems = new();
        private readonly Dictionary<string, bool> _categoryFoldStates = new();
        
        private ICheatService? _cheatService;
        private Vector2 _scrollPosition;

        [MenuItem(EditorConfig.ContextMenuWindowPath + "/Cheats")]
        public static void ShowWindow() => 
            GetWindow<CheatsEditorWindow>("WingPlay Cheats");

        private void OnEnable()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            if (EditorApplication.isPlaying)
                FindAndSubscribeToService().Forget();
        }

        private void OnDisable()
        {
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            UnsubscribeFromService();
        }

        private void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.EnteredPlayMode)
            {
                FindAndSubscribeToService().Forget();
            }
            else if (state == PlayModeStateChange.ExitingPlayMode)
            {
                UnsubscribeFromService();
                _cheatService = null;
                ClearCache();
                _categoryFoldStates.Clear();
            }
        }

        private void OnGUI()
        {
            if (_cheatService == null || !EditorApplication.isPlaying)
            {
                EditorGUILayout.HelpBox("Run Play Mode to see available cheats.", MessageType.Info);
                return;
            }


            _scrollPosition = EditorGUILayout.BeginScrollView(_scrollPosition);

            // Using .ToList() to safely modify the _categoryFoldStates dictionary while iterating
            foreach (var pair in _categorizedItems.ToList())
            {
                string categoryName = pair.Key;
                _categoryFoldStates.TryGetValue(categoryName, out bool isExpanded);

                bool newExpandedState = EditorGUILayout.Foldout(isExpanded, categoryName, true, EditorStyles.foldoutHeader);
                if (newExpandedState != isExpanded) 
                    _categoryFoldStates[categoryName] = newExpandedState;

                if (newExpandedState)
                {
                    EditorGUI.indentLevel++;
            
                    foreach (var item in pair.Value)
                    {
                        if (item is CheatActionInfo actionInfo)
                        {
                            if (GUILayout.Button(actionInfo.Name))
                                foreach (Action action in actionInfo.Actions) 
                                    action?.Invoke();
                        }
                        else if (item is CheatPropertyInfo propInfo)
                        {
                            using (new GUILayout.HorizontalScope())
                            {
                                EditorGUILayout.LabelField(propInfo.Name, GUILayout.Width(150));
                                DrawPropertyControl(propInfo);
                            }
                        }
                    }
            
                    EditorGUI.indentLevel--;
                }
            }

            EditorGUILayout.EndScrollView();
        }
        
        private void DrawPropertyControl(CheatPropertyInfo propInfo)
        {
            switch (propInfo.Property)
            {
                case BoolCheatProperty boolProp:
                    bool newValueBool = EditorGUILayout.Toggle(boolProp.Value);
                    if (GUI.changed) 
                        boolProp.ResetFromStringValue(newValueBool.ToString());
                    
                    break;
                case IntCheatProperty intProp:
                    int newValueInt = intProp is { Min: not null, Max: not null } 
                        ? EditorGUILayout.IntSlider(intProp.Value, intProp.Min.Value, intProp.Max.Value) 
                        : EditorGUILayout.IntField(intProp.Value);
                    if (GUI.changed) 
                        intProp.ResetFromStringValue(newValueInt.ToString());
                    
                    break;
                case FloatCheatProperty floatProp:
                    float newValueFloat = floatProp is { Min: not null, Max: not null }
                        ? EditorGUILayout.Slider(floatProp.Value, floatProp.Min.Value, floatProp.Max.Value)
                        : EditorGUILayout.FloatField(floatProp.Value);
                    if (GUI.changed) 
                        floatProp.ResetFromStringValue(newValueFloat.ToString(CultureInfo.InvariantCulture));
                    
                    break;
                case StringCheatProperty stringProp:
                    string newValueString = EditorGUILayout.TextField(stringProp.Value);
                    if (GUI.changed) 
                        stringProp.ResetFromStringValue(newValueString);
                    
                    break;
                case EnumCheatProperty enumProp:
                    string[] allValues = Enum.GetNames(enumProp.EnumType);
                    int newIndex = EditorGUILayout.Popup(Math.Max(0, Array.IndexOf(allValues, enumProp.Value)), allValues);
                    string newValue = allValues[newIndex];
                    if (GUI.changed) 
                        enumProp.ResetFromStringValue(newValue);
                    
                    break;
                case TimeSpanCheatProperty timeSpanProp:
                    TimeSpan timeSpan = timeSpanProp.Value;
                    string currentTimeSpanValueString = $"{(int)timeSpan.TotalDays}d {timeSpan.Hours}h {timeSpan.Minutes}m";
                    string newValueTimeSpan = EditorGUILayout.TextField(currentTimeSpanValueString);
                    if (GUI.changed) 
                        timeSpanProp.ResetFromStringValue(newValueTimeSpan);
                    
                    break;
                default:
                    EditorGUILayout.LabelField("Unsupported property type");
                    break;
            }
        }

        private async UniTask FindAndSubscribeToService()
        {
            while (_cheatService == null)
            {
                try { _cheatService ??= LifetimeScope.Find<LifetimeScope>().Container.Resolve<ICheatService>(); }
                catch { /* ignore */ }
                await Task.Delay(300);
            }
            
            SubscribeToService();
            RebuildCache();
        }

        private void SubscribeToService()
        {
            if (_cheatService == null) 
                return;

            _cheatService.ActionRegistered += OnCheatsChanged;
            _cheatService.ActionUnregistered += OnCheatsChanged;
            _cheatService.PropertyRegistered += OnCheatsChanged;
            _cheatService.PropertyUnregistered += OnCheatsChanged;
        }

        private void UnsubscribeFromService()
        {
            if (_cheatService == null) 
                return;

            _cheatService.ActionRegistered -= OnCheatsChanged;
            _cheatService.ActionUnregistered -= OnCheatsChanged;
            _cheatService.PropertyRegistered -= OnCheatsChanged;
            _cheatService.PropertyUnregistered -= OnCheatsChanged;
        }

        private void OnCheatsChanged(object info)
        {
            RebuildCache();
            Repaint();
        }

        private void RebuildCache()
        {
            ClearCache();
            if (_cheatService == null) 
                return;

            foreach (var action in _cheatService.Actions)
            {
                string category = action.Category ?? DefaultCategoryName;
                if (!_categorizedItems.ContainsKey(category)) 
                    _categorizedItems[category] = new List<object>();
                
                _categorizedItems[category].Add(action);
            }

            foreach (var prop in _cheatService.Properties)
            {
                string category = prop.Category ?? DefaultCategoryName;
                if (!_categorizedItems.ContainsKey(category)) 
                    _categorizedItems[category] = new List<object>();
                
                _categorizedItems[category].Add(prop);
            }

            // IsTestDevice property is always registered in the Player category
            // TODO: temporary disabled. Set it only from backoffice
            // {
            //     CheatPropertyInfo isTestDeviceProp = new CheatPropertyInfo
            //     {
            //         Name = Constants.Cheats.Properties.IsTestDevice, 
            //         Category = Constants.Cheats.Category.Player,
            //         Property = _cheatService.TestDevice
            //     };
            //     
            //     if (!_categorizedItems.ContainsKey(isTestDeviceProp.Category))
            //         _categorizedItems[isTestDeviceProp.Category] = new List<object>();
            //
            //     _categorizedItems[isTestDeviceProp.Category].Add(isTestDeviceProp);
            // }
        }

        private void ClearCache()
        {
            _categorizedItems.Clear();
        }
    }
}