import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyIntent, isFollowUp } from '../src/brain/intent.js';
import { detectLanguage } from '../src/brain/lang.js';

const aliases = ['sumit', 'soni', 'sumitscript'];
const intent = (t) => classifyIntent(t, { ownerAliases: aliases });

test('intent routing', () => {
  assert.equal(intent('hi'), 'greeting');
  assert.equal(intent('Good morning!'), 'greeting');
  assert.equal(intent('haha that was funny 😂'), 'joke');
  assert.equal(intent('why does my python code throw KeyError?'), 'coding');
  assert.equal(intent('where does Sumit work?'), 'about_owner');
  assert.equal(intent('what are your skills'), 'about_owner');
  assert.equal(intent('tumhara github kya hai'), 'about_owner');
  assert.equal(intent('who won the 2011 cricket world cup'), 'general');
  assert.equal(intent('12*7+3'), 'math');
  assert.equal(intent('thanks a lot'), 'thanks');
});

test('owner alias needs a word boundary', () => {
  assert.equal(intent('sonic the hedgehog trivia'), 'general');
});

test('follow-up detection', () => {
  assert.equal(isFollowUp('what about his second project?'), true);
  assert.equal(isFollowUp('Which programming languages does Sumit know?'), false);
});

test('language detection for Indian languages', () => {
  assert.equal(detectLanguage('भारत की राजधानी क्या है?').code, 'hi');
  assert.equal(detectLanguage('मला काय हवे आहे').code, 'mr');
  assert.equal(detectLanguage('તમે કેમ છો?').code, 'gu');
  assert.equal(detectLanguage('நீங்கள் எப்படி இருக்கிறீர்கள்').code, 'ta');
  assert.equal(detectLanguage('bhai kya scene hai aaj').code, 'hi-Latn');
  assert.equal(detectLanguage('What is the weather like today?').code, 'en');
  assert.equal(detectLanguage('I love the sun').code, 'en');
  assert.equal(detectLanguage('👍').code, 'und');
});
